# Load libraries
import numpy as np
import tensorflow as tf
tf.python.control_flow_ops = tf
import cv2
import matplotlib.pyplot as plt
import os
import csv
from random import shuffle
import sklearn

# Load the images in batches on the fly
samples = []
with open('./driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Split images into training and validation data
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Create the image generator function
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            correction_angle = 0.25 # Steering angle correction for left/right images
            for batch_sample in batch_samples:
                # Import center images
                name = './IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

##                # Flip center images
##                image_flipped = np.fliplr(center_image)
##                angle_flipped = -center_angle
##                images.append(image_flipped)
##                angles.append(angle_flipped)

                # Import left images
                name = './IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                left_angle = float(batch_sample[3]) + correction_angle
                images.append(left_image)
                angles.append(left_angle)

                # Import right images
                name = './IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3]) - correction_angle
                images.append(right_image)
                angles.append(right_angle)
                
            # Append imported images and steering angles
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Use the generator function to prepare images for compiling
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# Initial setup for Keras
from keras.models import Sequential, Model, load_model
from keras.layers import Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

# Keras model
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation="relu"))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation="relu"))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation="relu"))
model.add(Conv2D(64, 3, 3, subsample=(2,2), activation="relu"))
model.add(Conv2D(64, 3, 3, subsample=(2,2), activation="relu"))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile the model
model.compile(loss='mse', optimizer='adam')
print("Testing")
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples)*3, validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples)*3, 
    nb_epoch=7, verbose=1)

# Print the keys contained in the history object
print(history_object.history.keys())

# Plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

## Create HDF5 file 'model.h5'
model.save('model_NVIDIA.h5')
