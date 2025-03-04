import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# Paths
dataset_csv = "/home/riashad/projects/final-thesis/dataset/csv/final_dataset.csv"

# Parameters
batch_size = 32
image_size = (224, 224)
num_classes = 3
learning_rate = 1e-4
epochs = 30

# Load CSV and split data
df = pd.read_csv(dataset_csv)
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)

# Data Generators with Augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col="image_path",
    y_col="label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col="image_path",
    y_col="label",
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,  # Ensure proper evaluation
)

# Build the Model
base_model = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)  # Add dropout for regularization
x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)  # Add another dropout layer
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the Model
model.compile(
    optimizer=Adam(learning_rate=learning_rate),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
checkpoint = ModelCheckpoint(
    "DenseNet121.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)

early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=1)

callbacks = [checkpoint, early_stopping, lr_scheduler]


# Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1,
)

# Visualize Initial Training Learning Curves
plt.figure(figsize=(10, 6))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy (Initial Training)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (Initial Training)")
plt.legend()
plt.grid()
plt.show()

# Unfreeze and Fine-tune
for layer in base_model.layers:
    layer.trainable = True

# Recompile with lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Fine-tune
fine_tune_history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs // 2,
    callbacks=callbacks,
    verbose=1,
)

# Visualize Fine-tuning Learning Curves
plt.figure(figsize=(10, 6))
plt.plot(fine_tune_history.history["accuracy"], label="Train Accuracy (Fine-tune)")
plt.plot(
    fine_tune_history.history["val_accuracy"], label="Validation Accuracy (Fine-tune)"
)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy (Fine-tuning)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(fine_tune_history.history["loss"], label="Train Loss (Fine-tune)")
plt.plot(fine_tune_history.history["val_loss"], label="Validation Loss (Fine-tune)")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss (Fine-tuning)")
plt.legend()
plt.grid()
plt.show()

# Evaluation: Confusion Matrix and Classification Report
val_predictions = np.argmax(model.predict(val_generator), axis=-1)
val_labels = val_generator.classes  # Correctly retrieve true labels from the generator

cm = confusion_matrix(val_labels, val_predictions)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=val_generator.class_indices.keys()
)
disp.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix (Fine-tuning)")
plt.show()

report = classification_report(
    val_labels, val_predictions, target_names=val_generator.class_indices.keys()
)
print("Classification Report (Fine-tuning):\n", report)

# Save the Fine-tuned Model
model.save("DenseNet121_finetuned.keras")
