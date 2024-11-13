from datetime import datetime
import logging
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from glob import glob
import os
import cv2
from tqdm import tqdm

from transformers import ViTForImageClassification, TrainingArguments, Trainer

import torch
from torchvision import transforms
from torch.utils.data import Dataset

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
)


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
        return image, label, file_name


def calc_mean_std(train_df, trainloader):
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for input_image, _, _ in tqdm(trainloader):
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
df_train = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/train_df_imbalance_2020.csv"
)
df_valid = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/val_df_imbalance_2020.csv"
)
df_test = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/test_df_imbalance_2020.csv"
)

# Define the path to your train_images directory
train_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/train_images"

# Use glob to get all image files with .jpg or .jpeg extensions
image_files = glob(train_path + "/*.jp*g")

unique_labels = df_train.labels.value_counts()
num_unique_labels = unique_labels.index.nunique()


# Preprocessing
proc_resize = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(size=IMAGE_SIZE)]
)

train_df = ConstDataset(df_train, transform=proc_resize)

trainloader = torch.utils.data.DataLoader(
    train_df, batch_size, shuffle=True, num_workers=0
)

calc_mean = 0
calc_std = 0

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

proc_aug = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=calc_mean, std=calc_std),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.Normalize(mean=calc_mean, std=calc_std),
    ]
)


train_df = ConstDataset(df_train, transform=proc_aug)
valid_df = ConstDataset(df_valid, transform=proc_aug)
test_df = ConstDataset(df_test, transform=test_transforms)

dataloader = {
    "train": torch.utils.data.DataLoader(
        train_df, batch_size, shuffle=True, num_workers=0
    ),
    "val": torch.utils.data.DataLoader(
        valid_df, batch_size, shuffle=True, num_workers=0
    ),
    "test": torch.utils.data.DataLoader(
        test_df, batch_size, shuffle=True, num_workers=0
    ),
}

# Training model
model_name = "google/vit-base-patch16-224"
# model_name = "google/vit-huge-patch14-224-in21k"
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
    save_total_limit=2,  # Limit saved models to only the best and last
    load_best_model_at_end=True,  # Load the best model after training
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    logging_dir=os.path.join(output_dir, "logs"),  # Save logs in the timestamped folder
    remove_unused_columns=False,
    logging_steps=50,
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

val_image_ids = [item[2] for item in valid_df]
# Predict
outputs = trainer.predict(valid_df)
logger.info(f"Prediction metrics: {outputs.metrics}")
y_true = outputs.label_ids
y_pred = np.argmax(outputs.predictions, axis=1)

# Extract predicted probabilities from the last layer
val_probabilities = outputs.predictions  # This contains raw logits
val_probabilities = torch.softmax(
    torch.tensor(val_probabilities), dim=1
).numpy()  # Convert logits to probabilities

# Combine probabilities and labels into a DataFrame
val_prob_df = pd.DataFrame(
    val_probabilities,
    columns=[f"prob_class_{i}" for i in range(val_probabilities.shape[1])],
)
val_prob_df["image_id"] = val_image_ids

# Save to CSV
val_csv_path = os.path.join(output_dir, "validation_probabilities.csv")
val_prob_df.to_csv(val_csv_path, index=False)
logger.info(f"Validation probabilities saved at {val_csv_path}")
print(f"Validation probabilities saved at {val_csv_path}")

# # Confusion matrix
# labels = np.unique(train_df.labels)
# cm = confusion_matrix(y_true, y_pred, labels=labels)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

# # Save confusion matrix plot
# plt.figure(figsize=(10, 10))
# disp.plot(xticks_rotation=45)
# confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
# plt.savefig(confusion_matrix_path)
# logger.info(f"Confusion matrix saved at {confusion_matrix_path}")

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

# Predict on the test set
test_image_ids = [item[2] for item in test_df]
test_outputs = trainer.predict(test_df)
y_test_true = test_outputs.label_ids
y_test_pred = np.argmax(test_outputs.predictions, axis=1)

# Calculate probabilities for log loss
test_probabilities = torch.softmax(
    torch.tensor(test_outputs.predictions), dim=1
).numpy()

# Combine probabilities and labels into a DataFrame
test_prob_df = pd.DataFrame(
    test_probabilities,
    columns=[f"prob_class_{i}" for i in range(test_probabilities.shape[1])],
)
test_prob_df["image_id"] = test_image_ids

# Save to CSV
test_csv_path = os.path.join(output_dir, "test_probabilities.csv")
test_prob_df.to_csv(test_csv_path, index=False)
logger.info(f"Test probabilities saved at {test_csv_path}")
print(f"Test probabilities saved at {test_csv_path}")

# Calculate log loss
test_log_loss = log_loss(y_test_true, test_probabilities)
logger.info(f"Test Log Loss: {test_log_loss:.4f}")
print(f"Test Log Loss: {test_log_loss:.4f}")

# Calculate metrics
accuracy = accuracy_score(y_test_true, y_test_pred)
precision = precision_score(y_test_true, y_test_pred, average="weighted")
recall = recall_score(y_test_true, y_test_pred, average="weighted")
f1 = f1_score(y_test_true, y_test_pred, average="weighted")
f2 = fbeta_score(y_test_true, y_test_pred, beta=2, average="weighted")

# Log the metrics
logger.info(f"Test Accuracy: {accuracy:.4f}")
logger.info(f"Test Precision: {precision:.4f}")
logger.info(f"Test Recall: {recall:.4f}")
logger.info(f"Test F1 Score: {f1:.4f}")
logger.info(f"Test F2 Score: {f2:.4f}")

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print(f"Test F2 Score: {f2:.4f}")

# Confusion matrix for test set
test_cm = confusion_matrix(y_test_true, y_test_pred, labels=np.unique(y_test_true))
disp = ConfusionMatrixDisplay(
    confusion_matrix=test_cm, display_labels=np.unique(y_test_true)
)

# Save confusion matrix plot for test set
plt.figure(figsize=(10, 10))
disp.plot(xticks_rotation=45)
test_confusion_matrix_path = os.path.join(output_dir, "test_confusion_matrix.png")
plt.savefig(test_confusion_matrix_path)
logger.info(f"Test confusion matrix saved at {test_confusion_matrix_path}")
