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

BATCH_SIZE = 128

def set_image_size(new_image_size):
    global IMAGE_SIZE
    IMAGE_SIZE = new_image_size
    return IMAGE_SIZE

def set_batch_size(new_batch_size):
    global BATCH_SIZE
    BATCH_SIZE = new_batch_size
    return BATCH_SIZE

class TestConstDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        # self.labels = df["labels"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = os.path.join(train_path, file_name)
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        return image

# now we have the files, take the names and construct a dataframe of the file names
# we wanna access the pictures, take the pixels, and call calc_mean_std



def calc_mean_std(test_df, trainloader):
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    for input_image in tqdm(trainloader):
        psum += input_image.sum(axis=[0, 2, 3])
        psum_sq += (input_image**2).sum(axis=[0, 2, 3])

    count = len(test_df) * IMAGE_SIZE[0] * IMAGE_SIZE[1]
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean**2)
    total_std = torch.sqrt(total_var)

    mean = total_mean.numpy()
    std = total_std.numpy()
    return mean, std


#pass in dataframe, return calculated mean and standard deviation
def norm_data (dataframe):
    proc_resize = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize(size=IMAGE_SIZE)]
    )
    test_df = TestConstDataset(dataframe, transform = proc_resize)
    trainloader = torch.utils.data.DataLoader(
        test_df, BATCH_SIZE, shuffle=True, num_workers=0
    )
    calc_mean, calc_std = calc_mean_std(test_df, trainloader)
    print(f"mean = {calc_mean}, std = {calc_std}")

