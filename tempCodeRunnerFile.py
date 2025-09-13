import os
import sys
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONIOENCODING'] = 'UTF-8'
sys.stdout.reconfigure(encoding='utf-8')

print(os.getcwd())
print(os.listdir())
# import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import keras
# from tensorflow.keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model = keras.models.load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"


def getResult(img):
    try:
         image=cv2.imread(img)
         if image is None:
            raise ValueError(f"Failed to load image. Please check the path: {img}")
         image = Image.fromarray(image, 'RGB')
         image = image.resize((64, 64))
         image=np.array(image)
         input_img = np.expand_dims(image, axis=0)
         result = model.predict(input_img)
         return result
    except Exception as e:
         print(f"Error during prediction: {e}")
         return None


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        if value is not None:
            result = get_className(np.argmax(value))
            return result
        else:
            return "Error during prediction. Please try again."
        # result=get_className(value)
        # return result
    return None


if __name__ == '__main__':
    app.run(debug=True)