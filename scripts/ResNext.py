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
import torch.nn as nn
import torch.optim as optim

from sklearn import model_selection
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

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
    f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnext/{timestamp}/"
)
os.makedirs(output_dir, exist_ok=True)

# Create a logger
logging.basicConfig(
    filename=os.path.join(output_dir, "training.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Preparing Data
df_train_data = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/train_images_filtered_no_duplicates.csv"
)

# Define the path to your train_images directory
train_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/train_images"

# Use glob to get all image files with .jpg or .jpeg extensions
image_files = glob(train_path + "/*.jp*g")

# Data split
unique_labels = df_train_data.labels.value_counts()
num_unique_labels = unique_labels.index.nunique()
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
proc_aug = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=calc_mean, std=calc_std),
    ]
)

desired_majority_class_size = 6000

class_counts = df_train["labels"].value_counts()
undersample_strategy = {class_counts.idxmax(): desired_majority_class_size}

rus = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=109)
X_under, y_under = rus.fit_resample(
    df_train["image_id"].values.reshape(-1, 1), df_train["labels"].values
)

desired_minority_class_size = 6000

ros = RandomOverSampler(
    sampling_strategy={
        label: desired_minority_class_size
        for label in class_counts.index
        if class_counts[label] < desired_minority_class_size
    },
    random_state=109,
)
X_resampled, y_resampled = ros.fit_resample(X_under, y_under)

df_train_resampled = pd.DataFrame(
    {"image_id": X_resampled.flatten(), "labels": y_resampled}
)

df_train_resampled.reset_index(drop=True, inplace=True)

train_df = ConstDataset(df_train_resampled, transform=proc_aug)
valid_df = ConstDataset(df_valid, transform=proc_aug)

dataloader = {
    "train": torch.utils.data.DataLoader(
        train_df, batch_size, shuffle=True, num_workers=0
    ),
    "val": torch.utils.data.DataLoader(
        valid_df, batch_size, shuffle=True, num_workers=0
    ),
}

resnext = torch.hub.load("pytorch/vision:v0.10.0", "resnext50_32x4d", pretrained=True)

# set the number of classes for resnet18 model classification layer
num_classes = 5
# The input to the fully connected layer is resnet18.fc.in_features
resnext.fc = nn.Linear(resnext.fc.in_features, num_classes)

for name, param in resnext.named_parameters():
    if "fc" in name:  # Unfreeze the final classification layer
        param.requires_grad = True
    else:
        param.requires_grad = False

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(
    resnext.parameters(), lr=0.0005, momentum=0.9
)  # Use all parameters

# Move the model to the GPU if available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
resnext = resnext.to(device)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

logging.info("Start of training.")

for epoch in range(epochs):
    logging.info(f"Epoch {epoch+1}/{epochs}")
    logging.info("-" * 10)

    # Iterate over both training and validation phases
    for phase in ["train", "val"]:
        if phase == "train":
            resnext.train()  # Set the model to training mode
        else:
            resnext.eval()  # Set the model to evaluation mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data
        for inputs, labels in dataloader[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = resnext(inputs)
            with torch.set_grad_enabled(phase == "train"):
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass and optimization (if in training phase)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            # Update running loss and corrects
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        # Calculate epoch loss and accuracy
        if phase == "train":
            epoch_loss = running_loss / len(train_df)
            epoch_acc = running_corrects.double() / len(train_df)
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc.item())
        else:
            epoch_loss = running_loss / len(valid_df)
            epoch_acc = running_corrects.double() / len(valid_df)
            val_losses.append(epoch_loss)
            val_accuracies.append(epoch_acc.item())

        logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    logging.info("Epoch complete!")

# Save the model's state dictionary
save_path = os.path.join(output_dir, "trained_model.pth")
torch.save(resnext.state_dict(), save_path)
logging.info(f"Model saved to {save_path}")

# Plot and save the training and validation loss curves
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title("Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

# Plot and save the training and validation accuracy curves
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Accuracy Curves")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Save the figure
training_curves_path = os.path.join(output_dir, "training_curves.png")
plt.savefig(training_curves_path)
logging.info(f"Training curves saved at {training_curves_path}")

# Generate confusion matrix for validation data
resnext.eval()  # Set model to evaluation mode
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in dataloader["val"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnext(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
labels = sorted(set(all_labels))  # assuming labels are integers

# Display and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(10, 10))
disp.plot(xticks_rotation=45)
confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path)
logging.info(f"Confusion matrix saved at {confusion_matrix_path}")
