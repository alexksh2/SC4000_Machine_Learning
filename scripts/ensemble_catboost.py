import numpy as np
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import os
import joblib

# Set up directory and logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/ensemble_catboost/{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(output_dir, "ensemble_catboost.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

# List of CSV files containing individual model predictions
val_csv_files = [
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnext/20241110_145953_imbalance/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241110_134601_imbalance/validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnet/20241114_154207_imbalance/validation_inference_results.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cropnet/20241110_154159_imbalance/validation_inference_results.csv",
]

test_csv_files = [
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnext/20241110_145953_imbalance/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241110_134601_imbalance/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnet/20241114_154207_imbalance/test_inference_results.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cropnet/20241110_154159_imbalance/test_inference_results.csv",
]

valid_df = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/val_df_imbalance.csv"
)

test_df = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/test_df_imbalance.csv"
)

# Merge the CSV files
for i, file in enumerate(val_csv_files):
    df = pd.read_csv(file)

    if i == 0:
        merged_df = df
        # Move the image_id column to the first position
        image_name = merged_df.columns[-1]
        merged_df = merged_df[[image_name] + merged_df.columns[:-1].tolist()]

        # Merge with true labels
        merged_df = merged_df.merge(valid_df, on="image_id", how="left")
    else:
        # Merge subsequent prediction files
        merged_df = merged_df.merge(
            df, on="image_id", how="left", suffixes=("", f"_model{i+1}")
        )

# Extract features and labels
X_val = merged_df.drop(columns=["image_id", "labels"]).values
y_val = merged_df["labels"].values

# Define CatBoost parameters
catboost_params = {
    "iterations": 1000,
    "learning_rate": 0.01,
    "depth": 5,
    "loss_function": "MultiClass",
    "eval_metric": "MultiClass",
    "random_seed": 168,
    "verbose": 100,
    "early_stopping_rounds": 10,
}

# Create CatBoost Pool for validation data
train_pool = Pool(X_val, label=y_val)

# Train CatBoost model
catboost_model = CatBoostClassifier(**catboost_params)
catboost_model.fit(train_pool, verbose=True)

# Save the CatBoost model
model_path = os.path.join(output_dir, "ensemble_catboost_model.cbm")
catboost_model.save_model(model_path)
print(f"CatBoost model saved to {model_path}")

# Feature Importance
feature_importance = catboost_model.get_feature_importance()
feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importance}
).sort_values(by="Importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance for Ensemble Model")
plt.gca().invert_yaxis()

# Save the plot
plt.savefig(os.path.join(output_dir, "feature_importance.png"))

# # probabilities for val set
# val_probabilities = catboost_model.predict_proba(
#     X_val
# )  # Get probabilities for each class

# # Add the image_id and probabilities for each class to the DataFrame
# val_results_df = pd.DataFrame(
#     val_probabilities,
#     columns=[f"prob_class_{i}" for i in range(val_probabilities.shape[1])],
# )
# val_results_df["image_id"] = merged_df["image_id"].values

# # Save the DataFrame with probabilities and image_id
# val_results_path = os.path.join(
#     output_dir, "val_inference_results_with_probabilities.csv"
# )
# val_results_df.to_csv(val_results_path, index=False)
# print(f"Val inference results with probabilities saved at {val_results_path}")

# Merge the CSV files for test data
for i, file in enumerate(test_csv_files):
    df = pd.read_csv(file)

    if "Filename" in df.columns:
        df = df.rename(columns={"Filename": "image_id"})

    if i == 0:
        test_merged_df = df
        # Move the image_id column to the first position
        image_name = test_merged_df.columns[-1]
        test_merged_df = test_merged_df[
            [image_name] + test_merged_df.columns[:-1].tolist()
        ]

        # Merge with true labels
        test_merged_df = test_merged_df.merge(test_df, on="image_id", how="left")
    else:
        # Merge subsequent prediction files
        test_merged_df = test_merged_df.merge(
            df, on="image_id", how="left", suffixes=("", f"_model{i+1}")
        )

# Extract features and labels
X_test = test_merged_df.drop(columns=["image_id", "labels"]).values
y_test = test_merged_df["labels"].values

# Predict with the CatBoost meta-model on the test set
test_probabilities = catboost_model.predict_proba(
    X_test
)  # Get probabilities for each class
final_predictions = np.argmax(test_probabilities, axis=1)

# # Add the image_id and probabilities for each class to the DataFrame
# test_results_df = pd.DataFrame(
#     test_probabilities,
#     columns=[f"prob_class_{i}" for i in range(test_probabilities.shape[1])],
# )
# test_results_df["image_id"] = test_merged_df["image_id"].values

# # Save the DataFrame with probabilities and image_id
# test_results_path = os.path.join(
#     output_dir, "test_inference_results_with_probabilities.csv"
# )
# test_results_df.to_csv(test_results_path, index=False)
# print(f"Inference results with probabilities saved at {test_results_path}")

# Calculate metrics
accuracy = accuracy_score(y_test, final_predictions)
precision = precision_score(y_test, final_predictions, average="weighted")
recall = recall_score(y_test, final_predictions, average="weighted")
f1 = f1_score(y_test, final_predictions, average="weighted")
f2 = fbeta_score(y_test, final_predictions, beta=2, average="weighted")

# Log results
logger.info(f"Ensemble Test Accuracy: {accuracy:.8f}")
logger.info(f"Ensemble Test Precision: {precision:.8f}")
logger.info(f"Ensemble Test Recall: {recall:.8f}")
logger.info(f"Ensemble Test F1 Score: {f1:.8f}")
logger.info(f"Ensemble Test F2 Score: {f2:.8f}")

# Print logged metrics for reference
print(f"Accuracy: {accuracy:.8f}")
print(f"Precision: {precision:.8f}")
print(f"Recall: {recall:.8f}")
print(f"F1 Score: {f1:.8f}")
print(f"F2 Score: {f2:.8f}")

# Confusion Matrix
cm = confusion_matrix(y_test, final_predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(xticks_rotation=45)

# Save confusion matrix plot
plt.title("Confusion Matrix of CatBoost Ensemble Model")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()

# # Save the CatBoost model
# joblib.dump(catboost_model, os.path.join(output_dir, "ensemble_catboost_model.pkl"))
