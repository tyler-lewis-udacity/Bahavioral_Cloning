'''
To fix: GLib-GIO-Message: Using the 'memory' GSettings backend.  Your settings will not be saved or shared with other applications

Install the following in anaconda environment:
sudo apt-get install dconf-gsettings-backend
'''

import csv
import cv2
import numpy as np
import matplotlib as mp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import PIL
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


# Read in image and measurements data from .csv file:
def get_csv_data(csv_file='./data/driving_log.csv'):
    lines = []
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    return lines


lines = get_csv_data()


# TODO: convert to generator
def get_images():
    images = []
    for line in lines:
        for i in range(3):
            source_path = line[i]
            filename = source_path.split('/')[-1]
            current_path = './data/IMG/' + filename
            image = mpimg.imread(current_path)
            images.append(image)
    return images


images = get_images()


def get_measurements(correction=0.2):
    measurements = []
    for line in lines:
        steering_center = float(line[3])
        measurements.append(steering_center)
        steering_left = steering_center + correction
        measurements.append(steering_left)
        steering_right = steering_center - correction
        measurements.append(steering_right)

    return measurements


measurements = get_measurements()


# i = 4
# print(f'Steering = {measurements[i]}')
# img = images[i]
# plt.imshow(img)
# plt.show()


augmented_images = []
augmented_measurements = []

for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image, 1))
    augmented_measurements.append(measurement * -1.0)


# i = 3
# print(f'Steering = {augmented_measurements[i]}')
# img = augmented_images[i]
# plt.imshow(img)
# plt.show()


X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)


# # Simple Architecture:
# model = Sequential()
# model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# model.add(Flatten())
# model.add(Dense(1))

# NVIDIA Architecture:
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="elu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="elu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Convolution2D(64, 3, 3, activation="elu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')
