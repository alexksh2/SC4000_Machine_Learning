import os
import shutil
import pandas as pd

# Define paths
original_data_dir = "/home/samic_yongjian/tensorflow_datasets/downloads/extracted/ZIP.emcassavadata_cassavaleafdataiuY9Lvfe4ImdJSp0hSPmA_l-KucCLxA_kvxCqfznTZU.zip/cassavaleafdata/"
all_images_dir = (
    "/home/samic_yongjian/temp/SC4000_Machine_Learning/data/all_cassava_images"
)
os.makedirs(all_images_dir, exist_ok=True)

# Mapping of class names to label IDs
label_map = {"cbb": 0, "cbsd": 1, "cgm": 2, "cmd": 3, "healthy": 4}


# Helper function to process a split (train, test, validation)
def process_split(split_name):
    split_dir = os.path.join(original_data_dir, split_name)
    csv_data = []

    # Traverse each class folder in the split directory
    for class_name, label_id in label_map.items():
        class_dir = os.path.join(split_dir, class_name)

        # For each image file, copy to all_images_dir and record in csv_data
        for image_filename in os.listdir(class_dir):
            original_image_path = os.path.join(class_dir, image_filename)
            new_image_path = os.path.join(all_images_dir, image_filename)

            # Copy image to unified all_images_dir
            if not os.path.exists(new_image_path):  # Avoid duplicate copies
                shutil.copy2(original_image_path, new_image_path)

            # Append image_id and label to csv_data
            csv_data.append({"image_id": image_filename, "labels": label_id})

    # Save split data to CSV
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(f"{split_name}_labels.csv", index=False)
    print(f"{split_name}_labels.csv created with {len(csv_data)} entries.")


# Process each split
for split in ["train", "validation", "test"]:
    process_split(split)

print(
    "All images have been copied to 'all_cassava_images' and CSV files created for each split."
)
