import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Step 1: Set paths
data_dir = r"C:\Users\WALTON\Downloads\Datasets\Images\archive (1)\Data\Train_Data"
csv_file = r"C:\Users\WALTON\Downloads\Datasets\CSV\final_dataset.csv"

# Step 2: Load the CSV file with image paths and labels
df = pd.read_csv(csv_file)

# Step 3: Create a label map for categories
label_map = {'Nil': 0, 'Moderate': 1, 'Severe': 2}

# Step 4: Prepare data generators for loading images
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Step 5: Train-test split (80% training, 20% validation)
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'])

# Step 6: Use ImageDataGenerator to load images from directories
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",  # because we have more than 2 classes
    shuffle=True
)

val_generator = test_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col="image_path",
    y_col="label",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Step 7: Define the EfficientNetB3 model
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the base model
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(3, activation='softmax')  # 3 categories: Nil, Moderate, Severe
])

# Step 8: Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Step 9: Set up callbacks for early stopping and model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('fire_severity_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')

# Step 10: Train the model
history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)

# Step 11: Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")

# Step 12: Save the final model
model.save("final_fire_severity_model.h5")
