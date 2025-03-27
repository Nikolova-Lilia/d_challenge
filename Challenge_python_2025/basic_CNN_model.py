import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Define image size and paths
IMAGE_SIZE = (64, 64)  # Resize images
DATASET_PATH = "./emoji_dataset"  # Path where images are stored

# Load images and labels
def load_images():
    categories = ["happy", "sad", "crying", "surprised", "angry"]
    data, labels = [], []

    for label, category in enumerate(categories):
        folder_path = os.path.join(DATASET_PATH, category)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # Load in color
            img = cv2.resize(img, IMAGE_SIZE)  # Resize
            data.append(img)
            labels.append(label)

    return np.array(data, dtype="float32") / 255.0, np.array(labels)

# Load dataset
X, y = load_images()
y = keras.utils.to_categorical(y, num_classes=5)  # Convert labels to one-hot encoding

# Split into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(5, activation="softmax")  # 5 output classes
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

# Train model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the trained model
model.save("emoji_classifier.h5")
