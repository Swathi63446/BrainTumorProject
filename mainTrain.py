import cv2
import os
import tensorflow as tf
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

import keras

# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras.utils import to_categorical

image_directory = r'BrainTumor Classification DL\datasets'

#print("IMAGE DIRECTORY SUCCESS")

no_tumor_images = os.listdir(os.path.join(image_directory, 'no'))
yes_tumor_images = os.listdir(os.path.join(image_directory, 'yes'))
# print(no_tumor_images)
dataset=[]
label=[]

INPUT_SIZE=64
# print(no_tumor_images)

# path='no0.jpg'

# print(path.split('.')[1])



for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[-1].lower() == 'jpg':
        image_path = os.path.join(image_directory, 'no', image_name)
        # print(f"Reading image: {image_path}")
        image = cv2.imread(image_path)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(0)
        else:
            print(f"Warning: Could not read image {image_path}")




for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[-1].lower() == 'jpg':
        image_path = os.path.join(image_directory, 'yes', image_name)
        # print(f"Reading image: {image_path}")
        image = cv2.imread(image_path)
        if image is not None:
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            dataset.append(np.array(image))
            label.append(1)
        else:
            print(f"Warning: Could not read image {image_path}")


dataset=np.array(dataset)
label=np.array(label)


x_train, x_test, y_train, y_test=train_test_split(dataset, label, test_size=0.2, random_state=0)

# Reshape = (n, image_width, image_height, n_channel)

# print(x_train.shape)
# print(y_train.shape)

# print(x_test.shape)
# print(y_test.shape)

# Normalize pixel values to [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0


# x_train=normalize(x_train, axis=1)
# x_test=normalize(x_test, axis=1)

y_train = keras.utils.to_categorical(y_train , num_classes=2)
y_test = keras.utils.to_categorical(y_test , num_classes=2)



# Model Building
# 64,64,3

model= keras.models.Sequential()

model.add(keras.layers.Conv2D(32, (3,3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(32, (3,3), kernel_initializer='he_uniform'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))


model.add(keras.layers.Conv2D(64, (3,3), kernel_initializer='he_uniform'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(2))
model.add(keras.layers.Activation('softmax'))


# Binary CrossEntropy= 1, sigmoid
# Categorical Cross Entropy= 2 , softmax

model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])


model.fit(x_train, y_train, 
batch_size=16, 
verbose=1, epochs=10, 
validation_data=(x_test, y_test),
shuffle=False)


model.save('BrainTumor10EpochsCategorical.h5')

