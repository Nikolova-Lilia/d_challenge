import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tkinter as tk
from tkinter import filedialog
from tensorflow.keras.models import load_model

def detect_emojis(image_path):
    model = keras.models.load_model("emoji_classifier.h5")
    class_names = ["happy", "sad", "crying", "surprised", "angry"]
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    detected_emojis = []

    for y in range(0, height - 64, 32):  # Move in steps of 32 pixels
        for x in range(0, width - 64, 32):
            roi = image[y:y+64, x:x+64]  # Extract region of interest (ROI)
            roi = cv2.resize(roi, (64, 64)) / 255.0  # Normalize
            roi = np.expand_dims(roi, axis=0)

            prediction = model.predict(roi)
            emoji_type = class_names[np.argmax(prediction)]

            if np.max(prediction) > 0.7:  # Confidence threshold
                detected_emojis.append((emoji_type, (x, y)))

    # Print the required output
    print(f"Picture: {image_path}")
    for emoji, coord in detected_emojis:
        print(f"Emoji: {emoji} Coordinates: {coord}")

# Example usage
detect_emojis("test_images/emoji_0.jpg")


# Load trained model
model = load_model("emoji_classifier.h5")

def load_and_predict():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    img = cv2.resize(img, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_names = ["happy", "sad", "crying", "surprised", "angry"]
    result_label.config(text=f"Prediction: {class_names[np.argmax(predictions)]}")

# Create GUI
root = tk.Tk()
root.title("emoji Classifier")

btn = tk.Button(root, text="Upload Image", command=load_and_predict)
btn.pack()

result_label = tk.Label(root, text="Prediction: ")
result_label.pack()

# Print the required output
print(f"Picture: {image_path}")
for emoji, coord in detected_emojis:
    print(f"Emoji: {emoji} Coordinates: {coord}")
root.mainloop()
