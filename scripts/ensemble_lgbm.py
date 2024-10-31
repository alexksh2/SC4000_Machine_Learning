import numpy as np
import lightgbm as lgb
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

# Sample individual model predictions (replace with actual data)
cnn_val_preds = np.random.rand(100, 10)
alexnet_val_preds = np.random.rand(100, 10)
efficientnet_val_preds = np.random.rand(100, 10)
inception_val_preds = np.random.rand(100, 10)
resnet_val_preds = np.random.rand(100, 10)
resnext_val_preds = np.random.rand(100, 10)
vit_val_preds = np.random.rand(100, 10)

# Stack validation predictions and labels
X_val = np.hstack(
    (
        cnn_val_preds,
        alexnet_val_preds,
        efficientnet_val_preds,
        inception_val_preds,
        resnet_val_preds,
        resnext_val_preds,
        vit_val_preds,
    )
)
y_val = np.random.randint(0, 10, 100)  # Replace with actual validation labels

# Set up LightGBM dataset
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
    "n_estimators": 10000,
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
fold_accuracies = []
cv_results = lgb.cv(
    params,
    train_data,
    num_boost_round=100,
    nfold=5,
    stratified=True,
    shuffle=True,
    metrics="multi_logloss",
    early_stopping_rounds=10,
    verbose_eval=10,
    seed=42,
    callbacks=[lgb.record_evaluation(fold_accuracies)],  # Track accuracy for each fold
)

# Accuracy for each fold
for i, acc in enumerate(fold_accuracies):
    logger.info(f"Fold {i+1} Accuracy: {acc['multi_logloss-mean'][-1]:.4f}")
    print(f"Fold {i+1} Accuracy: {acc['multi_logloss-mean'][-1]:.4f}")

# Best number of boosting rounds
best_num_boost_round = len(cv_results["multi_logloss-mean"])
logger.info(f"Best number of boosting rounds from CV: {best_num_boost_round}")

# Train final model with optimal rounds
gbm = lgb.train(params, train_data, num_boost_round=best_num_boost_round)

# Feature Importance
feature_importance = gbm.feature_importance()
feature_names = [f"Feature_{i}" for i in range(X_val.shape[1])]
importance_df = pd.DataFrame(
    {"Feature": feature_names, "Importance": feature_importance}
)
importance_df = importance_df.sort_values(by="Importance", ascending=False)

# Log feature importance
logger.info("Feature Importance:")
for index, row in importance_df.iterrows():
    logger.info(f"{row['Feature']}: {row['Importance']}")
    print(f"{row['Feature']}: {row['Importance']}")

# Plot feature importance
plt.figure(figsize=(10, 8))
plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance for Ensemble Model")
plt.gca().invert_yaxis()

# Save the plot
plt.savefig(os.path.join(output_dir, "feature_importance.png"))

# Test data
cnn_test_preds = np.random.rand(50, 10)
alexnet_test_preds = np.random.rand(50, 10)
efficientnet_test_preds = np.random.rand(50, 10)
inception_test_preds = np.random.rand(50, 10)
resnet_test_preds = np.random.rand(50, 10)
resnext_test_preds = np.random.rand(50, 10)
vit_test_preds = np.random.rand(50, 10)

# Stack test predictions
X_test = np.hstack(
    (
        cnn_test_preds,
        alexnet_test_preds,
        efficientnet_test_preds,
        inception_test_preds,
        resnet_test_preds,
        resnext_test_preds,
        vit_test_preds,
    )
)

# Predict with the LightGBM meta-model on the test set
final_predictions = np.argmax(gbm.predict(X_test), axis=1)

# True labels for test set
y_test = np.random.randint(0, 10, 50)  # Replace with actual test labels

# Calculate metrics
accuracy = accuracy_score(y_test, final_predictions)
precision = precision_score(y_test, final_predictions, average="weighted")
recall = recall_score(y_test, final_predictions, average="weighted")
f1 = f1_score(y_test, final_predictions, average="weighted")
f2 = fbeta_score(y_test, final_predictions, beta=2, average="weighted")

# Log results
logger.info(f"Ensemble Test Accuracy: {accuracy:.4f}")
logger.info(f"Ensemble Test Precision: {precision:.4f}")
logger.info(f"Ensemble Test Recall: {recall:.4f}")
logger.info(f"Ensemble Test F1 Score: {f1:.4f}")
logger.info(f"Ensemble Test F2 Score: {f2:.4f}")

# Print logged metrics for reference
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"F2 Score: {f2:.4f}")

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
