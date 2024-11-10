from datetime import datetime
import logging
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from glob import glob
import os
import cv2
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim import AdamW
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    f1_score,
)

import timm
from torch.nn.utils import clip_grad_norm_

# IMAGE CONFIGURATIONS
IMAGE_SIZE = [128, 128]

# TRAINING CONFIGURATIONS
epochs = 20
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
logger = logging.getLogger()

# Preparing Data
df_train = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/train_df_imbalance.csv"
)
df_valid = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/val_df_imbalance.csv"
)
df_test = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/test_df_imbalance.csv"
)

# Define the path to your train_images directory
train_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/train_images_old"

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
proc_aug = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(mean=calc_mean, std=calc_std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=IMAGE_SIZE),
        transforms.Normalize(mean=calc_mean, std=calc_std),
    ]
)

train_df = ConstDataset(df_train, transform=proc_aug)
valid_df = ConstDataset(df_valid, transform=proc_aug)
test_df = ConstDataset(df_test, transform=test_transform)


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

# Hyperparameters for training loop
max_grad_norm = 1000
num_classes = 5
criterion = nn.CrossEntropyLoss()

# class for custom RexNext
class CustomResNext(nn.Module):
    def __init__(self, model_name='resnext50_32x4d', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, num_classes)

    def forward(self, x):
        x = self.model(x)
        return x

# Initialize model and optimizer
resnext = CustomResNext()
optimizer = AdamW(resnext.parameters(), lr=1e-4, weight_decay=1e-6)
scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.2, patience = 5, verbose = True, eps = 1e-6)

# Move the model to the GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnext = resnext.to(device)

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []

best_val_loss = float("inf")
best_val_probs = None  # To store the best validation probabilities
best_val_labels = None  # To store the corresponding true labels
best_model_path = os.path.join(output_dir, "best_model.pth")

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
        all_probs = []
        all_labels = []
        all_ids = []

        # Iterate over data
        for inputs, labels, filenames in dataloader[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = resnext(inputs)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            with torch.set_grad_enabled(phase == "train"):
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # Backward pass and optimization (if in training phase)
                if phase == "train":
                    loss.backward()
                    grad_norm = clip_grad_norm_(resnext.parameters(), max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()

            if phase == "val":
                probs = torch.softmax(outputs, dim=1)
                all_probs.extend(probs.detach().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_ids.extend(filenames)

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

            scheduler.step(epoch_loss)

            if epoch_loss < best_val_loss:
                best_val_loss = epoch_loss
                best_val_probs = all_probs  # Save the best probabilities
                best_val_labels = all_labels  # Save the corresponding true labels
                best_image_ids = all_ids
                # Save best model
                torch.save(resnext.state_dict(), best_model_path)
                logging.info(
                    f"New best model found at epoch {epoch+1} with validation loss {best_val_loss:.4f}"
                )
                logging.info(f"Best model saved at {best_model_path}")

        logging.info(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    logging.info("Epoch complete!")

# Save the model's state dictionary
save_path = os.path.join(output_dir, "last.pth")
torch.save(resnext.state_dict(), save_path)
logging.info(f"Model saved to {save_path}")

# Save the best validation probabilities to CSV
best_val_df = pd.DataFrame(
    best_val_probs, columns=[f"prob_class_{i}" for i in range(len(best_val_probs[0]))]
)
best_val_df["image_id"] = best_image_ids
best_val_csv_path = os.path.join(output_dir, "best_validation_probabilities.csv")
best_val_df.to_csv(best_val_csv_path, index=False)
logging.info(f"Best validation probabilities saved at {best_val_csv_path}")
print(f"Best validation probabilities saved at {best_val_csv_path}")

# Plot and save the training and validation loss curves
plt.figure(figsize=(12, 6))

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
all_probs = []
all_filenames = []

with torch.no_grad():
    for inputs, labels, filenames in dataloader["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = resnext(inputs)

        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Get predicted probabilities (after softmax)
        probs = torch.softmax(outputs, dim=1)
        all_probs.extend(probs.cpu().numpy())

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        all_filenames.extend(filenames)


test_prob_df = pd.DataFrame(
    all_probs, columns=[f"prob_class_{i}" for i in range(len(all_probs[0]))]
)
test_prob_df["Filename"] = all_filenames
test_csv_path = os.path.join(output_dir, "test_probabilities.csv")
test_prob_df.to_csv(test_csv_path, index=False)
logging.info(f"Test probabilities saved at {test_csv_path}")
print(f"Test probabilities saved at {test_csv_path}")

# Compute evaluation metrics
logloss = log_loss(all_labels, all_probs)
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average="weighted")
recall = recall_score(all_labels, all_preds, average="weighted")
f1 = f1_score(all_labels, all_preds, average="weighted")
f2 = fbeta_score(all_labels, all_preds, beta=2, average="weighted")

# Log the metrics
logging.info(f"Log Loss: {logloss:.4f}")
logging.info(f"Accuracy: {accuracy:.4f}")
logging.info(f"Precision: {precision:.4f}")
logging.info(f"Recall: {recall:.4f}")
logging.info(f"F1 Score: {f1:.4f}")
logging.info(f"F2 Score: {f2:.4f}")

print(f"Log Loss: {logloss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"F2 Score: {f2:.4f}")

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
labels = sorted(set(all_labels))  # assuming labels are integers

# Display and save the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
plt.figure(figsize=(10, 10))
disp.plot(xticks_rotation=45)
test_confusion_matrix_path = os.path.join(output_dir, "test_confusion_matrix.png")
plt.savefig(test_confusion_matrix_path)
logging.info(f"Confusion matrix saved at {test_confusion_matrix_path}")
