import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# Function to load and preprocess images for the model
def load_and_preprocess_image(file_path, target_size):
    img = load_img(
        file_path, target_size=target_size
    )  # Resize based on model's input size
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return img_array


# Function to test bulk images with a model and print category counts
def test_bulk_images(
    image_folder_path, model, label_encoder, severity_categories, target_size
):
    severity_count = {
        category: 0 for category in severity_categories
    }  # Initialize counters

    # Iterate over all images in the folder
    for filename in os.listdir(image_folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(image_folder_path, filename)

            # Load and preprocess the image
            image = load_and_preprocess_image(file_path, target_size)
            image = np.expand_dims(image, axis=0)  # Add batch dimension

            # Predict severity
            severity_score = model.predict(image)[0]
            severity_class = np.argmax(severity_score)
            severity_label = label_encoder.inverse_transform([severity_class])[0]

            # Update the counter for the predicted severity category
            severity_count[severity_label] += 1

    # Print the counts for each severity category
    for category, count in severity_count.items():
        print(f"{category}: {count}")


# Function to load model dynamically
def load_model_by_name(model_name):
    model_path = os.path.join(os.path.dirname(__file__), "..", f"{model_name}.h5")
    return load_model(model_path)


# Main function
def main():
    # Define severity categories
    severity_categories = ["nil", "low", "mid", "high", "deadly"]

    # Initialize the label encoder and fit it with the severity categories
    label_encoder = LabelEncoder()
    label_encoder.fit(severity_categories)

    # Define model names and corresponding target sizes
    models_info = {
        "vgg16": {"model": "VGG16", "target_size": (64, 64)},
        "resnet50": {"model": "ResNet50", "target_size": (64, 64)},
    }

    # Define the image folder paths for testing
    folders = {
        # "nil": r"C:\Users\WALTON\Downloads\Large-Fire\non-fire",
        # "low": r"C:\Users\WALTON\Downloads\Large-Fire\Fire-2",
        # "mid": r"C:\Users\WALTON\Downloads\Large-Fire\Fire-4",
        # "high": r"C:\Users\WALTON\Downloads\Large-Fire\Fire-7",
        "deadly": r"C:\Users\WALTON\Downloads\Large-Fire\Fire-9",
    }

    # Loop through models and categories
    for model_name, model_info in models_info.items():
        print(f"\nTesting {model_name.upper()} Model:")

        # Load the model dynamically
        model = load_model_by_name(model_info["model"])

        # Test images for each category
        for category, folder_path in folders.items():
            print(f"Testing category: {category}")
            test_bulk_images(
                folder_path,
                model,
                label_encoder,
                severity_categories,
                target_size=model_info["target_size"],
            )


# Run the script
if __name__ == "__main__":
    main()
