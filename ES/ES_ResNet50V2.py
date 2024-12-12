import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Paths
train_dir = r"C:\Users\WALTON\Downloads\Datasets\Images\archive (1)\Data\Train_Data"
batch_size = 32
img_size = (224, 224)

# Data Augmentation for Training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2,
)

# Training and Validation Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
)

# Load Pre-Trained ResNet50V2 with Fine-Tuning
base_model = ResNet50V2(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # Freeze base model initially

# Add Custom Top Layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)  # Regularization
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation="softmax")(x)

# Build Model
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Callbacks
checkpoint = ModelCheckpoint(
    "best_resnet50v2_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
)

early_stop = EarlyStopping(
    monitor="val_accuracy", patience=10, restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
)

callbacks = [checkpoint, early_stop, lr_scheduler]

# Initial Training with Frozen Base
print("Training with frozen base layers...")
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15,
    callbacks=callbacks,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
)

# Fine-Tuning: Unfreeze some layers in ResNet50V2
print("Fine-tuning deeper layers...")
for layer in base_model.layers[-50:]:  # Unfreeze last 50 layers
    layer.trainable = True

# Recompile model with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Fine-Tune Training
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=35,  # Extend training
    callbacks=callbacks,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
)

# Save Final Model
model.save("ResNet50V2.keras")
print("Model training complete!")
