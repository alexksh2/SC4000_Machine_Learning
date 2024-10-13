from datetime import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os
import cv2
from tqdm import tqdm

from transformers import ViTForImageClassification, TrainingArguments, Trainer

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim import AdamW

from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# IMAGE CONFIGURATIONS
IMAGE_SIZE = [224, 224]

# TRAINING CONFIGURATIONS
epochs = 30
batch_size = 128


class ConstDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        self.labels = df["labels"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(train_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image, label


def calc_mean_std(train_df, trainloader):
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for input_image, _ in tqdm(trainloader):
        psum += input_image.sum(axis=[0, 2, 3])
        psum_sq += (input_image**2).sum(axis=[0, 2, 3])

    count = len(train_df) * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    mean = total_mean.numpy()
    std = total_std.numpy()
    return mean, std


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = (
    f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/{timestamp}/"
)
os.makedirs(output_dir, exist_ok=True)

# Create a logger
logging.basicConfig(
    filename=os.path.join(output_dir, "training.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

# Preparing Data
df_train_data = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/merged_train.csv"
)

# Define the path to your train_images directory
train_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/train_images"

# Use glob to get all image files with .jpg or .jpeg extensions
image_files = glob(train_path + "/*.jp*g")

# Data split
unique_labels = df_train_data.labels.value_counts()
num_unique_labels = unique_labels.nunique()
df_train, df_valid = model_selection.train_test_split(
    df_train_data,
    test_size=0.2,
    random_state=109,
    stratify=df_train_data["labels"].values,
)
df_train.reset_index(drop=True, inplace=True)
df_valid.reset_index(drop=True, inplace=True)

# Preprocessing
proc_resize = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(size=IMAGE_SIZE)]
)

train_df = ConstDataset(df_train, transform=proc_resize)

trainloader = torch.utils.data.DataLoader(
    train_df, batch_size, shuffle=True, num_workers=0
)


calc_mean, calc_std = calc_mean_std(train_df, trainloader)

# Data augmentation
train_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.Normalize(mean=calc_mean, std=calc_std),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.Normalize(mean=calc_mean, std=calc_std),
    ]
)

train_df = ConstDataset(df_train, transform=train_transforms)
valid_df = ConstDataset(df_valid, transform=val_transforms)

dataloader = {
    "train": torch.utils.data.DataLoader(
        train_df, batch_size, shuffle=True, num_workers=0
    ),
    "val": torch.utils.data.DataLoader(
        valid_df, batch_size, shuffle=True, num_workers=0
    ),
}

# Training model
model_name = "google/vit-base-patch16-224"
num_classes = num_unique_labels

model = ViTForImageClassification.from_pretrained(
    model_name, num_labels=num_classes, ignore_mismatched_sizes=True
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)


# Training arguments configuration
args = TrainingArguments(
    output_dir=output_dir,  # Use timestamped folder
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir=os.path.join(output_dir, "logs"),  # Save logs in the timestamped folder
    remove_unused_columns=False,
)


# Collate function for combining samples into batches
def collate_fn(examples):
    images = torch.stack([example[0] for example in examples])  # Stack images
    labels = torch.tensor([example[1] for example in examples])  # Stack labels
    return {"pixel_values": images, "labels": labels}


# Metric computation (accuracy)
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    logger.info(f"Validation accuracy: {accuracy:.4f}")
    return {"accuracy": accuracy}


# Initialize the trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_df,
    eval_dataset=valid_df,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Predict
outputs = trainer.predict(valid_df)
logger.info(f"Prediction metrics: {outputs.metrics}")
y_true = outputs.label_ids
y_pred = np.argmax(outputs.predictions, axis=1)

# Confusion matrix
labels = np.unique(train_df.labels)
cm = confusion_matrix(y_true, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# Save confusion matrix plot
plt.figure(figsize=(10, 10))
disp.plot(xticks_rotation=45)
confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
logger.info(f"Confusion matrix saved at {confusion_matrix_path}")

# Plot training & validation loss/accuracy
history = trainer.state.log_history
train_losses = [log["loss"] for log in history if "loss" in log]
eval_accuracies = [log["eval_accuracy"] for log in history if "eval_accuracy" in log]

plt.figure(figsize=(12, 6))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Plot validation accuracy
plt.subplot(1, 2, 2)
plt.plot(eval_accuracies, label="Validation Accuracy", color="orange")
plt.title("Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

# Save plots
training_plot_path = os.path.join(output_dir, "training_plots.png")
plt.savefig(training_plot_path)
logger.info(f"Training plots saved at {training_plot_path}")

# Optional: Launch tensorboard
# %load_ext tensorboard
# %tensorboard --logdir ../output/logs
