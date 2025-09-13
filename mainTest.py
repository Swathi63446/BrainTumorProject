import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'UTF-8'

import cv2
#from keras.models import load_model # type: ignore
from PIL import Image
import numpy as np
import keras
import tensorflow as tf

# Ensure stdout uses utf-8 encoding
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Load the Keras model
model = keras.models.load_model('BrainTumor10EpochsCategorical.h5')

# Read the image
image=cv2.imread('C:/Users/LENOVO/Desktop/mini project/BrainTumor Classification DL/uploads/pred0.jpg')


if image is None:
    print(f"Failed to load image. Please check the path: ")
else:
    print(f"Image loaded successfully. Shape: {image.shape}")

# Convert the image to a PIL Image
img=Image.fromarray(image)

img=img.resize((64,64))

# Convert the PIL Image to a numpy array
img=np.array(img)

# Expand dimensions to match the input shape of the model (batch size, height, width, channels)
input_img=np.expand_dims(img, axis=0)

# # Make a prediction
# # prediction = model.predict(input_img)
# prediction = model.predict(input_img)
# print(f"Prediction: {prediction}")

# Make a prediction
try:
    prediction = model.predict(input_img)
    print(f"Prediction: {prediction}")

    # Get the predicted class
    predicted_class = np.argmax(prediction, axis=1)
    print(predicted_class)
except Exception as e:
    print(f"Error during prediction: {e}")



# Get the predicted class
predicted_class = np.argmax(prediction, axis=1)
print(predicted_class)




