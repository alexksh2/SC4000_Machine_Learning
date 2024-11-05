import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
import logging
import os
import joblib
import pandas as pd
from datetime import datetime

# Set up directory and logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"/home/samic_yongjian/temp/SC4000_Machine_Learning/output/ensemble_lightgbm_kfold/{timestamp}/"
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(output_dir, "ensemble_lightgbm.log"),
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

# Load validation and test data
valid_df = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/valid_df.csv"
)
test_df = pd.read_csv(
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/test_df.csv"
)

# List of CSV files containing individual model predictions
val_csv_files = [
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnext/20241103_232814/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241103_190631/validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit_v2/20241104_164221/validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnet/20241104_010113/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/inception/20241104_124742/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4/20241103_215449/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4_v2/20241104_004159/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb0/20241104_125022/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cnn/20241104_143543/best_validation_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/alexnet/20241104_143840/best_validation_probabilities.csv",
]

test_csv_files = [
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnext/20241103_232814/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit/20241103_190631/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/vit_v2/20241104_164221/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/resnet/20241104_010113/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/inception/20241104_124742/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4/20241103_215449/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb4_v2/20241104_004159/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/efficientnetb0/20241104_125022/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/cnn/20241104_143543/test_probabilities.csv",
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/output/alexnet/20241104_143840/test_probabilities.csv",
]

# Merge the CSV files
for i, file in enumerate(val_csv_files):
    df = pd.read_csv(file)
    if i == 0:
        merged_df = df
        image_name = merged_df.columns[-1]
        merged_df = merged_df[[image_name] + merged_df.columns[:-1].tolist()]
        merged_df = merged_df.merge(valid_df, on="image_id", how="left")
    else:
        merged_df = merged_df.merge(
            df, on="image_id", how="left", suffixes=("", f"_model{i+1}")
        )

# Extract features and labels
X_val = merged_df.drop(columns=["image_id", "labels"]).values
y_val = merged_df["labels"].values

# LightGBM parameters
params = {
    "objective": "multiclass",
    "num_class": 5,
    "metric": "multi_error",
    "boosting_type": "gbdt",
    "learning_rate": 0.01,
    "min_data_in_leaf": 20,
    "num_leaves": 10,
    "max_depth": 5,
    "subsample": 0.3,
    "colsample_bytree": 0.3,
    "reg_alpha": 0.15,
    "reg_lambda": 0.15,
    "verbose": -1,
    "n_jobs": 2,
    "random_state": 168,
}

# Perform KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []
best_iter_count = []

for fold_idx, (train_index, valid_index) in enumerate(kf.split(X_val)):
    X_train_fold, X_valid_fold = X_val[train_index], X_val[valid_index]
    y_train_fold, y_valid_fold = y_val[train_index], y_val[valid_index]

    train_data_fold = lgb.Dataset(X_train_fold, label=y_train_fold)
    valid_data_fold = lgb.Dataset(
        X_valid_fold, label=y_valid_fold, reference=train_data_fold
    )

    evals_result_fold = {}
    gbm_fold = lgb.train(
        params,
        train_data_fold,
        num_boost_round=1000,
        valid_sets=[train_data_fold, valid_data_fold],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100),
            lgb.record_evaluation(evals_result_fold),
        ],
    )

    # Store the evaluation results for this fold
    fold_results.append(evals_result_fold)
    logger.info(
        f"Fold {fold_idx + 1} completed with best iteration: {gbm_fold.best_iteration}"
    )
    best_iter_count.append(gbm_fold.best_iteration)

# Log and print evaluation results for each fold
# for fold_idx, fold_result in enumerate(fold_results):
#     print(f"\nFold {fold_idx + 1} Evaluation Results:")
#     logger.info(f"\nFold {fold_idx + 1} Evaluation Results:")
# for metric_name, metric_values in fold_result.items():
#     print(
#         f"  {metric_name} (valid): Last recorded value = {metric_values['valid'][-1]:.4f}"
#     )
#     logger.info(
#         f"  {metric_name} (valid): Last recorded value = {metric_values['valid'][-1]:.4f}"
#     )

best_iterations = np.mean(best_iter_count)

# Calculate the average best iteration
final_best_iteration = int(np.mean(best_iterations))
logger.info(f"Final number of boosting rounds for training: {final_best_iteration}")

# Train the final model using all data and the average best iteration
final_gbm = lgb.train(
    params,
    lgb.Dataset(X_val, label=y_val),
    num_boost_round=1000,
)

# Feature Importance
feature_importance = final_gbm.feature_importance()
feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importance}
)
importance_df = importance_df.sort_values(by="Importance", ascending=False)


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
final_predictions = np.argmax(final_gbm.predict(X_test), axis=1)

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

# Save the final model
joblib.dump(final_gbm, os.path.join(output_dir, "ensemble_lightgbm_final_model.pkl"))
logger.info("Final model saved.")
