import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

model_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models/fire_severity_model_transfer_learning.h5",
)
model_severity = tf.keras.models.load_model(model_path)


# Function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(64, 64))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(
        img_array, axis=0
    )  # Expand dimensions to match model input shape
    return img_array


# Example image path for testing (replace with your own test image path)
cur = os.path.dirname(__file__)
test_image_path = os.path.join(cur, "..", "test_media/small_camp_fire.jpg")

# Load and preprocess the test image
test_image = load_and_preprocess_image(test_image_path)

# Predict severity score
severity_score = model_severity.predict(test_image)[0][0]
# severity_score = model_severity.predict(test_image)
print(f"Predicted Severity Score: {severity_score}")
