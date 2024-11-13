import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping
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
output_dir = f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/ensemble_lightgbm/{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(output_dir, "ensemble_lightgbm.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

# List of CSV files containing individual model predictions
val_csv_files = [
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnext/20241113_164038 (m1920_on_d20)/validation_inference_results.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241113_171759 (m1920_on_d20)/validation_inference_results.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit_v2/20241104_164221/validation_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnet/20241104_010113/best_validation_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/inception/20241104_124742/best_validation_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4/20241103_215449/best_validation_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4_v2/20241104_004159/best_validation_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb0/20241104_125022/best_validation_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cnn/20241104_143543/best_validation_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/alexnet/20241104_143840/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnet/20241113_220413_2020/validation_inference_results.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cropnet/20241112_115704_2020/validation_inference_results.csv",
]

test_csv_files = [
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnext/20241113_164038 (m1920_on_d20)/test_inference_results.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241113_171759 (m1920_on_d20)/test_inference_results.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit_v2/20241104_164221/test_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnet/20241104_010113/test_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/inception/20241104_124742/test_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4/20241103_215449/test_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4_v2/20241104_004159/test_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb0/20241104_125022/test_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cnn/20241104_143543/test_probabilities.csv",
    # "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/alexnet/20241104_143840/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnet/20241113_220413_2020/test_inference_results.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cropnet/20241112_115704_2020/test_inference_results.csv",
]

valid_df = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/val_df_imbalance_2020.csv"
)

test_df = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/test_df_imbalance_2020.csv"
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

# Convert to LightGBM dataset format
train_data = lgb.Dataset(X_val, label=y_val)

# LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "min_data_in_leaf": 20,
    "verbose": -1,
    "num_leaves": 10,
    "max_depth": 5,
    "subsample": 0.3,
    "colsample_bytree": 0.3,
    "reg_alpha": 0.15,
    "reg_lambda": 0.15,
    "silent": True,
    "n_jobs": 2,
    "random_state": 168,
}

# Perform 5-fold cross-validation with LightGBM
evals_result = {}
cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=1000,
    nfold=5,
    stratified=True,
    shuffle=True,
    metrics="multi_logloss",
    seed=42,
    callbacks=[
        early_stopping(stopping_rounds=10),
        lgb.record_evaluation(evals_result),
    ],  # Track accuracy for each fold
)

# # Print or log detailed cross-validation results
# print("Detailed cross-validation results:")
# for metric_name, metric_results in evals_result.items():
#     for key, values in metric_results.items():
#         print(f"{key}: Last recorded value = {values[-1]:.4f}")
#         logger.info(f"{key}: Last recorded value = {values[-1]:.4f}")


# Train final model with optimal rounds
gbm = lgb.train(params, train_data, 600)

# Feature Importance
feature_importance = gbm.feature_importance()
feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importance}
)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Log feature importance
# logger.info("Feature Importance:")
# for index, row in importance_df.iterrows():
#     logger.info(f"{row['Feature']}: {row['Importance']}")
#     print(f"{row['Feature']}: {row['Importance']}")

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance for Ensemble Model")
plt.gca().invert_yaxis()

# Save the plot
plt.savefig(os.path.join(output_dir, "feature_importance.png"))

# Merge the CSV files
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

# Predict with the LightGBM meta-model on the test set
final_predictions = np.argmax(gbm.predict(X_test), axis=1)

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
plt.title("Confusion Matrix of LightGBM Ensemble Model (Cross-Validated)")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.show()

# Save the LightGBM model
joblib.dump(gbm, os.path.join(output_dir, "ensemble_lightgbm_model.pkl"))
