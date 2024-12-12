# Paths
input_folder = r"/home/repliq/Desktop/final-thesis/dataset/images/augment/Moderate"  # Folder with original moderate images
output_folder = r"/home/repliq/Desktop/final-thesis/dataset/images/augment/adv-augmented-moderate"  # Folder to save augmented images
import albumentations as A
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img, save_img

# Define the augmentation pipeline (without Cutout)
transform = A.Compose(
    [
        A.Rotate(limit=20),  # Rotate by up to 20 degrees
        A.RandomSizedCrop(min_max_height=(20, 30), height=224, width=224, p=0.5),
        A.HorizontalFlip(p=0.5),  # Flip horizontally with 50% probability
        A.RandomBrightnessContrast(p=0.2),  # Adjust brightness and contrast
        A.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5
        ),  # Random color jitter
    ]
)


# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process images
num_augmented_per_image = 3  # Number of new images to generate per original

for image_file in os.listdir(input_folder):
    if image_file.endswith((".jpg", ".png", ".jpeg")):  # Check for valid image formats
        # Load image
        img_path = os.path.join(input_folder, image_file)
        img = cv2.imread(img_path)  # Load image using OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Apply augmentation
        augmented = transform(image=img)
        augmented_image = augmented["image"]

        # Save augmented image
        augmented_img = array_to_img(augmented_image)
        save_img(os.path.join(output_folder, f"aug_{image_file}"), augmented_img)

print("Augmentation complete! Augmented images are saved in:", output_folder)
