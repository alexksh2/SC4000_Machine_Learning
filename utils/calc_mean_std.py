import pandas as pd
from glob import glob
import os
import cv2
from tqdm import tqdm

import torch
from torchvision import transforms
from torch.utils.data import Dataset


# IMAGE CONFIGURATIONS
IMAGE_SIZE = [128, 128]

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


# Preparing Data
df_train = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/train_labels.csv"
)

# Define the path to your train_images directory
train_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/all_cassava_images"

# Use glob to get all image files with .jpg or .jpeg extensions
image_files = glob(train_path + "/*.jp*g")


# Preprocessing
proc_resize = transforms.Compose(
    [transforms.ToTensor(), transforms.Resize(size=IMAGE_SIZE)]
)

train_df = ConstDataset(df_train, transform=proc_resize)

trainloader = torch.utils.data.DataLoader(
    train_df, batch_size, shuffle=True, num_workers=0
)


calc_mean, calc_std = calc_mean_std(train_df, trainloader)

print(f"mean = {calc_mean}, std = {calc_std}")
