import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load and preprocess images
def load_images(before_path, after_path, size=(128, 128)):
    before_img = cv2.imread(before_path)
    after_img = cv2.imread(after_path)

    before_img = cv2.resize(before_img, size) / 255.0
    after_img = cv2.resize(after_img, size) / 255.0

    return before_img, after_img

# Simple CNN model (binary classifier)
def create_model(input_shape):
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Simulated training (you can replace this with real data)
def simulate_training(model):
    # Generate synthetic forest and deforested data (for example purposes only)
    X = np.random.rand(20, 128, 128, 3)
    y = np.random.randint(0, 2, 20)
    model.fit(X, y, epochs=5, verbose=1)
    return model

# Prediction and visualization
def detect_deforestation(before_img, after_img, model):
    diff = np.abs(after_img - before_img)
    diff_input = np.expand_dims(diff, axis=0)
    prediction = model.predict(diff_input)[0][0]

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.title("Before")
    plt.imshow(before_img)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("After")
    plt.imshow(after_img)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"Deforestation: {round(prediction * 100)}%")
    plt.imshow(diff)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Main workflow
if _name_ == "_main_":
    # Replace with your image paths
    before_path = "before.jpg"
    after_path = "after.jpg"

    if not os.path.exists(before_path) or not os.path.exists(after_path):
        print("Place 'before.jpg' and 'after.jpg' in the same directory as this script.")
    else:
        before_img, after_img = load_images(before_path, after_path)
        model = create_model((128, 128, 3))
        model = simulate_training(model)  # Simulated; replace with real training in production
        detect_deforestation(before_img, after_img, model)