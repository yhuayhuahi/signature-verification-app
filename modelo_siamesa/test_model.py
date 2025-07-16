import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the size of the input images
from utils import SIZE

from keras.config import enable_unsafe_deserialization
enable_unsafe_deserialization()

# Load the trained model
model = tf.keras.models.load_model("siamese_signature_model.keras")

# Function to preprocess a single image (resize and normalize)
def preprocess_single_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, target_size)
    img = img.astype('float32') / 255.0
    return np.expand_dims(img, axis=-1)  # Add channel dimension

# Function to verify if two signatures match
def verify_signature(signature1_path, signature2_path):
    # Preprocess both images
    img1 = preprocess_single_image(signature1_path)
    img2 = preprocess_single_image(signature2_path)
    
    # Display images using matplotlib
    original = cv2.imread(signature1_path)
    forged = cv2.imread(signature2_path)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Signature")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(forged, cmap='gray')
    plt.title("Compared Signature")
    plt.axis('off')
    
    plt.show()
    
    # Make prediction using the Siamese network
    prediction = model.predict([np.expand_dims(img1, axis=0), np.expand_dims(img2, axis=0)])
    
    # If the output is closer to 1, it is a genuine match, otherwise, it is a forged signature
    result = "Genuine" if prediction > 0.8 else "Forged"
    print(f"The signature comparison result is: {result}")
    print(f"Similarity score: {prediction[0][0]}")

# Paths to the test signature images
signature1_path = "../dataset/signatures/full_org/original_10_1.png"
signature2_path = "../dataset/signatures/full_forg/forgeries_10_1.png"

# Run the signature verification
verify_signature(signature1_path, signature2_path)
