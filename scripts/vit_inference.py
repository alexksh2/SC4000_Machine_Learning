from datetime import datetime
import torch
import timm
from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification, AutoConfig
import pandas as pd
import numpy as np
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
import matplotlib.pyplot as plt
import os
from PIL import Image

# Paths and configurations
IMAGE_SIZE = [128, 128]
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = (
    f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/{timestamp}/"
)
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vit_output_model_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241110_134601_imbalance/checkpoint-559/model.safetensors"  # Update with the actual path

num_classes = 5

# Custom Dataset for ResNext
class CassavaDataset(Dataset):
    def __init__(self, dataframe, data_path, transform=None):
        self.dataframe = dataframe
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = os.path.join(self.data_path, self.dataframe.iloc[idx]["image_id"])
        image = Image.open(image_path).convert("RGB")
        label = self.dataframe.iloc[idx]["labels"]
        if self.transform:
            image = self.transform(image)
        return image, label


transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.4309022, 0.49720255, 0.31429148],
            std=[0.22041768, 0.22340892, 0.21165892],
        ),
    ]
)

# Load model configuration and model weights
config = AutoConfig.from_pretrained(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241110_134601_imbalance/checkpoint-559"
)  # Path to the folder with config.json
vit_model = ViTForImageClassification.from_pretrained(
    vit_output_model_path, config=config, local_files_only=True
)

vit_model.to(device)
vit_model.eval()

# Inference and evaluation function for ResNeXt
def run_vit_inference(dataset_df, dataset_name):
    dataset = CassavaDataset(dataset_df, data_path, transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Perform inference
    pred_probs = []
    true_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        with torch.no_grad():
            outputs = vit_model(images)
            probabilities = torch.softmax(outputs.logits, dim=1).cpu().numpy().flatten()
            pred_probs.append(probabilities)
            true_labels.append(labels.item())

    # Convert probabilities to predicted labels
    pred_labels = np.argmax(pred_probs, axis=1)

    # Evaluation metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")
    f1 = f1_score(true_labels, pred_labels, average="weighted")
    f2 = fbeta_score(true_labels, pred_labels, beta=2, average="weighted")
    logloss = log_loss(true_labels, pred_probs)
    print(f"{dataset_name} - Accuracy: {accuracy:.4f}")
    print(f"{dataset_name} - Precision: {precision:.4f}")
    print(f"{dataset_name} - Recall: {recall:.4f}")
    print(f"{dataset_name} - F1 Score: {f1:.4f}")
    print(f"{dataset_name} - F2 Score: {f2:.4f}")
    print(f"{dataset_name} - Log Loss: {logloss:.4f}")

    # Save results to CSV
    results_df = pd.DataFrame(
        pred_probs, columns=[f"prob_class_{i}" for i in range(num_classes)]
    )
    results_df["image_id"] = dataset_df["image_id"].values
    results_path = os.path.join(output_dir, f"{dataset_name}_inference_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"{dataset_name} inference results saved at {results_path}")

    # Confusion Matrix
    cm = confusion_matrix(true_labels, pred_labels)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    confusion_matrix_path = os.path.join(
        output_dir, f"{dataset_name}_confusion_matrix.png"
    )
    plt.savefig(confusion_matrix_path)
    print(f"{dataset_name} confusion matrix saved at {confusion_matrix_path}")


# Running inference
data_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/train_images"
df_valid = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/val_df_imbalance_2020.csv"
)
df_test = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/test_df_imbalance_2020.csv"
)

run_vit_inference(df_valid, "validation")
run_vit_inference(df_test, "test")
