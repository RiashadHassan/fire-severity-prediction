import os
from tensorflow.keras.preprocessing.image import (
    load_img,
    img_to_array,
    save_img,
    ImageDataGenerator,
)

# Paths
input_folder = r"C:\Users\WALTON\Downloads\Datasets\Images\archive (1)\Data\Train_Data\Moderate"  # Folder with original moderate images
output_folder = r"C:\Users\WALTON\Downloads\Datasets\Images\archive (1)\Data\Train_Data\augmented"  # Folder to save augmented images

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Augmentation settings
datagen = ImageDataGenerator(
    rotation_range=20,  # Rotate images up to 20 degrees
    width_shift_range=0.2,  # Shift image width by up to 20% of total width
    height_shift_range=0.2,  # Shift image height by up to 20% of total height
    brightness_range=[0.8, 1.2],  # Adjust brightness by ±20%
    zoom_range=0.2,  # Zoom in/out by 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    fill_mode="nearest",  # Fill pixels outside boundaries with nearest values
)

# Process images
num_augmented_per_image = 3  # Number of new images to generate per original

for image_file in os.listdir(input_folder):
    if image_file.endswith((".jpg", ".png", ".jpeg")):  # Check for valid image formats
        # Load and preprocess image
        img_path = os.path.join(input_folder, image_file)
        img = load_img(img_path)  # Load image as PIL object
        img_array = img_to_array(img)  # Convert to NumPy array
        img_array = img_array.reshape(
            (1,) + img_array.shape
        )  # Reshape for the generator

        # Generate augmented images
        i = 0
        for batch in datagen.flow(
            img_array,
            batch_size=1,
            save_to_dir=output_folder,
            save_prefix="moderate",
            save_format="jpeg",
        ):
            i += 1
            if i >= num_augmented_per_image:
                break  # Stop after generating the desired number of augmentations

print("Augmentation complete! Augmented images are saved in:", output_folder)
