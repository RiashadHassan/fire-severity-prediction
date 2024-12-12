# import os
# import pandas as pd

# # Set the base directory for the images
# data_dir = "/home/repliq/Desktop/final-thesis/dataset/images"

# # Initialize an empty list to store image paths and labels
# image_data = []

# # Iterate through the three folders (Moderate, Nil, Severe)
# for category in ["Severe", "Moderate", "Nil"]:
#     category_dir = os.path.join(data_dir, category)

#     # Ensure the category folder exists
#     if os.path.exists(category_dir):
#         for root, _, files in os.walk(category_dir):
#             for file in files:
#                 # Get the full path of the image
#                 image_path = os.path.join(root, file)

#                 # Add the image path and category label to the list
#                 image_data.append([image_path, category])

# # Create a DataFrame from the list of image data
# df = pd.DataFrame(image_data, columns=["image_path", "label"])

# # Save the DataFrame to a CSV file
# output_csv_path = r"/home/repliq/Desktop/final-thesis/dataset/csv/linux_dataset.csv"
# df.to_csv(output_csv_path, index=False)

# print(f"CSV file with image paths and labels saved to: {output_csv_path}")


import pandas as pd

# Path to the input CSV file
input_csv = r"/home/repliq/Desktop/final-thesis/dataset/csv/linux_dataset.csv"

# Path to the output CSV file (shuffled)
output_csv = r"/home/repliq/Desktop/final-thesis/dataset/csv/final_linux_dataset.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(input_csv)

# Shuffle the rows of the DataFrame
shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the shuffled DataFrame back to a new CSV file
shuffled_df.to_csv(output_csv, index=False)

print(f"Shuffled CSV saved to: {output_csv}")
