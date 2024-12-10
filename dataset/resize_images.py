import os
from PIL import Image

# Define paths
input_csv = r"C:\Users\WALTON\Downloads\Datasets\CSV\final_dataset.csv"
output_dir = r"C:\Users\WALTON\Downloads\Datasets\Images\archive (1)\Data\Train_Data\final"
target_size = (224, 224)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read the CSV file
import pandas as pd

df = pd.read_csv(input_csv)

# Iterate through each image in the dataset
for i, row in df.iterrows():
    input_path = row["image_path"]
    label = row["label"]

    # Create label subfolder in output directory
    label_dir = os.path.join(output_dir, label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Resize and save image
    try:
        img = Image.open(input_path)
        img = img.convert("RGB")  # Ensure image is in RGB format
        img = img.resize(target_size)
        output_path = os.path.join(label_dir, os.path.basename(input_path))
        img.save(output_path)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

print(f"All images resized and saved in: {output_dir}")
