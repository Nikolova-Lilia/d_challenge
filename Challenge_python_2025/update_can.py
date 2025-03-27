import cv2
import time
import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

# Load pre-trained emoji classifier model
model = keras.models.load_model("emoji_classifier.h5")
class_names = ["happy", "sad", "crying", "surprised", "angry"]

# Define scanning box size
BOX_SIZE = 32
scan_speed = 2  # Initial scan speed (milliseconds delay)

# Folder containing test images
TEST_IMAGES_FOLDER = "test_images"
image_files = sorted([f for f in os.listdir(TEST_IMAGES_FOLDER) if f.endswith(('.jpg', '.png'))])
image_index = 0  # Start with the first image

def is_emoji_pixel(pixel):
    """Detects if a pixel belongs to an emoji (not white)."""
    return not (pixel[0] > 200 and pixel[1] > 200 and pixel[2] > 200)

def classify_emoji(roi):
    """Classify an emoji in a given image region using the CNN model."""
    roi = cv2.resize(roi, (64, 64)) / 255.0  # Resize & normalize
    roi = np.expand_dims(roi, axis=0)  # Reshape for model input
    prediction = model.predict(roi)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    
    if confidence > 0.7:  # Confidence threshold
        return class_names[class_index], confidence
    return None, None  # If confidence is too low

def process_image(image_path):
    """Process an image to detect and classify emojis."""
    global scan_speed

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}!")
        return []

    cv2.namedWindow("Emoji Scanner", cv2.WINDOW_AUTOSIZE)
    display_img = img.copy()
    detected_emojis = []
    emoji_boxes = []  # To store coordinates of potential emojis

    # Scan the image using a grid approach
    for y in range(0, img.shape[0] - BOX_SIZE, BOX_SIZE):
        for x in range(0, img.shape[1] - BOX_SIZE, BOX_SIZE):
            contains_emoji = False
            roi = img[y:y+BOX_SIZE, x:x+BOX_SIZE]

            # Find non-white pixels to detect the emoji area
            for sy in range(y, min(y + BOX_SIZE, img.shape[0])):
                for sx in range(x, min(x + BOX_SIZE, img.shape[1])):
                    if is_emoji_pixel(img[sy, sx]):
                        contains_emoji = True
                        break

            if contains_emoji:
                # Store the coordinates of the non-white pixels (emoji area)
                emoji_boxes.append((x, y, x + BOX_SIZE, y + BOX_SIZE))

                # Classify the emoji in the detected region
                emoji_type, confidence = classify_emoji(roi)
                if emoji_type:
                    detected_emojis.append((emoji_type, (x, y)))
                    cv2.putText(display_img, emoji_type, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.rectangle(display_img, (x, y), (x + BOX_SIZE, y + BOX_SIZE), (0, 255, 0), 2)

            # Display scanning animation (Red box indicating current scan)
            cv2.rectangle(display_img, (x, y), (x + BOX_SIZE, y + BOX_SIZE), (0, 0, 255), 2)
            cv2.imshow("Emoji Scanner", display_img)
            time.sleep(scan_speed / 1000.0)
            key = cv2.waitKey(1) & 0xFF

            # Exit or adjust scanning speed
            if key == 27 or cv2.getWindowProperty("Emoji Scanner", cv2.WND_PROP_VISIBLE) < 1:
                return detected_emojis
            if key == ord('d'):
                scan_speed = min(scan_speed + 5, 100)
                print(f"Scan speed increased: {scan_speed} ms")
            if key == ord('i'):
                scan_speed = max(scan_speed - 5, 1)
                print(f"Scan speed decreased: {scan_speed} ms")

    # Now draw a blue bounding box around the area where emojis were detected
    for (x1, y1, x2, y2) in emoji_boxes:
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show the final image with blue boxes around detected emojis
    cv2.imshow("Emoji Scanner", display_img)
    cv2.waitKey(0)  # Wait for a key press before closing the window
    cv2.destroyAllWindows()

    return detected_emojis

def main():
    global image_index

    while True:
        if image_index >= len(image_files):
            print("No more images to process!")
            break

        image_path = os.path.join(TEST_IMAGES_FOLDER, image_files[image_index])
        print(f"Processing: {image_files[image_index]}")

        detected_emojis = process_image(image_path)

        # Print the results in the required format
        print(f"Picture: {image_files[image_index]}")
        for emoji, coord in detected_emojis:
            print(f"Emoji: {emoji} Coordinates: {coord}")

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC to exit
                cv2.destroyAllWindows()
                return
            elif key == ord('n') or key == 83:  # Right arrow key → next image
                image_index = min(image_index + 1, len(image_files) - 1)
                break
            elif key == ord('p') or key == 81:  # Left arrow key ← previous image
                image_index = max(image_index - 1, 0)
                break

if __name__ == "__main__":
    main()
