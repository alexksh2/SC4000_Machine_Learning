import pandas as pd
from sklearn.model_selection import train_test_split

# Load the data
input_csv_path = "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020/train.csv"  # Update this path
data = pd.read_csv(input_csv_path)

# Check if columns are named correctly
if not {"image_id", "labels"}.issubset(data.columns):
    raise ValueError("The input CSV must contain 'image_id' and 'labels' columns.")

# Perform an initial stratified split to separate 80% training data and 20% (val+test) data
train_data, temp_data = train_test_split(
    data, test_size=0.2, stratify=data["labels"], random_state=42
)

# Perform a secondary stratified split on the 20% temp data to get 10% validation and 10% test data
val_data, test_data = train_test_split(
    temp_data, test_size=0.5, stratify=temp_data["labels"], random_state=42
)

# Print the size of each split
print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")

# Save each split to CSV files
output_dir = (
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data_2020"  # Update this path
)
train_data.to_csv(f"{output_dir}/train_df_imbalance_2020.csv", index=False)
val_data.to_csv(f"{output_dir}/val_df_imbalance_2020.csv", index=False)
test_data.to_csv(f"{output_dir}/test_df_imbalance_2020.csv", index=False)

print("Data splits saved successfully:")
print(f"- Train: {output_dir}/train_df_imbalance_2020.csv")
print(f"- Validation: {output_dir}/val_df_imbalance_2020.csv")
print(f"- Test: {output_dir}/test_df_imbalance_2020.csv")
