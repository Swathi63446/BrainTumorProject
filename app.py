import os
import sys
import numpy as np
from PIL import Image
import cv2
import keras
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Set environment variables for TensorFlow and Python encoding
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'UTF-8'
sys.stdout.reconfigure(encoding='utf-8')

print(os.getcwd())  # Print the current working directory
print(os.listdir())  # List all files in the current directory

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Function to get class name based on the model prediction
def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"

# Function to process the image and make a prediction
def getResult(img_path):
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Failed to load image. Please check the path: {img_path}")
        
        image = Image.fromarray(image, 'RGB')
        image = image.resize((64, 64))
        image = np.array(image)
        input_img = np.expand_dims(image, axis=0)
        
        result = model.predict(input_img)
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# Route to render the index.html template
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to handle the image upload and prediction
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        value = getResult(file_path)
        if value is not None:
            result = get_className(np.argmax(value))
            return result
        else:
            return "Error during prediction. Please try again."
    return None

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
