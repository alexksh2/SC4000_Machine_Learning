from datetime import datetime
import pandas as pd
import cv2
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

# Image and path configurations
IMAGE_SIZE = [224, 224]  # CropNet model expects 224x224 images
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = (
    f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/{timestamp}/"
)
os.makedirs(output_dir, exist_ok=True)


