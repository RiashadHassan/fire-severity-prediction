import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


def test_bulk_images(image_folder_path):
    # Specify the path to the trained model
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "models/fire_severity_model_transfer_learning.h5",
    )
    print(f"Loading model from {model_path}")

    # Load the trained model
    model = load_model(model_path)

    # Define severity categories
    severity_categories = ["nil", "low", "mid", "high", "deadly"]

    # Initialize the label encoder and fit it with the severity categories
    label_encoder = LabelEncoder()
    label_encoder.fit(severity_categories)

    # Function to load and preprocess images
    def load_and_preprocess_image(file_path):
        # Resize image to match model input size (128x128 for transfer learning models like VGG16, etc.)
        img = load_img(file_path, target_size=(64, 64))
        img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
        return img_array

    # Initialize a counter for each severity category
    severity_count = {category: 0 for category in severity_categories}

    # Iterate over all images in the folder
    for filename in os.listdir(image_folder_path):
        # Check if the file is an image file
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(image_folder_path, filename)
            print(f"Processing image: {filename}")

            # Load and preprocess the image
            image = load_and_preprocess_image(file_path)
            image = np.expand_dims(
                image, axis=0
            )  # Expand dimensions to match model input shape

            # Predict severity
            severity_score = model.predict(image)[0]
            severity_class = np.argmax(severity_score)
            severity_label = label_encoder.inverse_transform([severity_class])[0]

            print(f"Predicted severity for {filename}: {severity_label}")

            # Update the counter for the predicted severity category
            severity_count[severity_label] += 1

    # Print the counts for each severity category
    print("\nSeverity counts:")
    for category, count in severity_count.items():
        print(f"{category}: {count}")


# Folder paths for images
nil = r"C:\Users\WALTON\Downloads\Large-Fire\non-fire"
low = r"C:\Users\WALTON\Downloads\Large-Fire\Fire-2"
mid = r"C:\Users\WALTON\Downloads\Large-Fire\Fire-4"
high = r"C:\Users\WALTON\Downloads\Large-Fire\Fire-7"
deadly = r"C:\Users\WALTON\Downloads\Large-Fire\Fire-9"

# Test the function with the folder paths
print("Testing on 'high' severity images:")
test_bulk_images(high)

print("\nTesting on 'low' severity images:")
# test_bulk_images(low)

print("\nTesting on 'mid' severity images:")
# test_bulk_images(mid)

print("\nTesting on 'deadly' severity images:")
# test_bulk_images(deadly)

print("\nTesting on 'nil' (non-fire) images:")
# test_bulk_images(nil)
