import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
import cv2
from keras import backend as K
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from sklearn.metrics import confusion_matrix
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
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


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Image and path configurations
IMAGE_SIZE = [224, 224]  # CropNet model expects 224x224 images
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = (
    f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficient_b4_final/{timestamp}/"
)
os.makedirs(output_dir, exist_ok=True)
# Load the pretrained CropNet model

model = tf.keras.models.load_model('../checkpoint/inception.keras')

WIDTH = 224
HEIGHT = 224    

# Preprocessing function
def preprocess_image(image_path):
    image = tf.keras.preprocessing.image.load_img(image_path)
    resized_image = image.resize((WIDTH, HEIGHT))
    return resized_image

# Inference and evaluation function with probability redistribution before argmax
def run_inference(dataset_df, dataset_name):
    image_ids = dataset_df["image_id"].values
    true_labels = dataset_df["labels"].values
    # Perform inference
    pred_probs = []
    for image_name in image_ids:
        image_path = os.path.join(data_path, image_name)
        image = preprocess_image(image_path)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = tf.cast(image, tf.float32)
        probabilities = model.predict(image)
        #probabilities = probabilities.numpy().flatten()  # Flatten to a 6-class vector
        # Check if the probability for class 5 exists
        if len(probabilities) > 5:
            # Extract the extra probability for class 5 and the first 5 classes
            extra_prob = probabilities[5]
            valid_probs = probabilities[:5]
            # Redistribute the extra probability proportionally across classes 0-4
            total_valid_prob = sum(valid_probs)
            redistributed_probs = [
                prob + (extra_prob * (prob / total_valid_prob)) for prob in valid_probs
            ]
        else:
            # No extra class, use the original probabilities
            redistributed_probs = probabilities[:5]
        pred_probs.append(redistributed_probs)
    # Convert redistributed probabilities to predicted labels
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
    # Save probabilities and predictions to CSV
    results_df = pd.DataFrame(pred_probs, columns=[f"prob_class_{i}" for i in range(5)])
    results_df["image_id"] = image_ids
    # results_df["predicted_label"] = pred_labels
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

# Paths and datasets
data_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/all_cassava_images"
df_valid = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/validation_labels.csv"
)
df_test = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/test_labels.csv"
)

# Run inference on validation and test sets
run_inference(df_valid, "validation")
run_inference(df_test, "test")
# df_testing = pd.read_csv(
#     "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/merged_train.csv"
# )
# run_inference(df_testing, "testing")