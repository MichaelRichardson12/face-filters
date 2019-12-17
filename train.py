import numpy as np
import matplotlib.pyplot as plt
import math
import cv2
from PIL import Image
import time
from utils import *

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


EPOCHS = 15


# Load training set
X_train, y_train = load_data()
print("X_train.shape == {}".format(X_train.shape))
print("y_train.shape == {}; y_train.min == {:.3f}; y_train.max == {:.3f}".format(
    y_train.shape, y_train.min(), y_train.max()))

# Load testing set
X_test, _ = load_data(test=True)
print("X_test.shape == {}".format(X_test.shape))


fig = plt.figure(1, figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_train[i], y_train[i], ax)
fig.savefig('generated/preview_images.png')


model = Sequential()

model.add(Convolution2D(16, 3, 3, input_shape=(96, 96, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Convolution2D(8, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(4, 3, 3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(30))

# Summarize the model
model.summary()


# optimizers = ['sgd','rmsprop','adagrad','adadelta','adam','adamax','nadam']
# best = 100
# best_optimizer = ""
# for optr in optimizers:
#     print(optr)
#     model.compile(loss='mean_squared_error', optimizer=optr)
#     hist = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=8, verbose=1)
#     last = hist.history['val_loss'][-1]
#     if last < best:
#         best = last
#         best_optimizer = optr
# print('Best optimiser: ', best_optimizer)


# Compile, fit and save the model using the best optimizer
model.compile(loss='mean_squared_error', optimizer='adamax')
hist = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=10, verbose=1)
model.save('face_feature_model.h5')


plt.figure(2)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('training and validation loss')
plt.ylabel('mean squared error')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.savefig('generated/model_loss.png')


y_test = model.predict(X_test)

fig = plt.figure(3, figsize=(20,20))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(9):
    ax = fig.add_subplot(3, 3, i + 1, xticks=[], yticks=[])
    plot_data(X_test[i], y_test[i], ax)
fig.savefig('generated/prediction.png')


image = cv2.imread('images/obamas4.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 2, 3)

print('Number of faces detected:', len(faces))

image_with_detections = np.copy(image)


size = 60
kernel = np.ones((size, size),np.float32)/(size*size)
# Get the bounding box for each detected face
for (x,y,w,h) in faces:
    # Add a red bounding box to the detections image
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 3)
    resized = cv2.resize(gray[y:y+h, x:x+w] ,(96,96))
    resized_color = cv2.resize(image[y:y+h, x:x+w], (96,96))
    X = np.vstack(resized) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1) # return each images as 96 x 96 x 1
    keypoints = model.predict(X)[0]
    keypoints = keypoints * 48 + 48 # undo the normalization
    keyx = keypoints[0::2]
    keyy = keypoints[1::2]
    for kx, ky in zip(keyx, keyy):
        cv2.circle(resized_color, (kx,ky), 1, (0,255,0), 1)
    image[y:y+h, x:x+w] = cv2.resize(resized_color, (h,w))

# Display the image with the detections
fig = plt.figure(4, figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Image with Face Detections')
ax1.imshow(image)
fig.savefig('generated/test_obamas.png')


sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)

# Plot the image
fig = plt.figure(5, figsize = (6,6))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.imshow(sunglasses)
ax1.axis('off');

print ('The sunglasses image has shape: ' + str(np.shape(sunglasses)))

# Print out the sunglasses transparency (alpha) channel
alpha_channel = sunglasses[:,:,3]
print ('the alpha channel here looks like')
print (alpha_channel)

# Just to double check that there are indeed non-zero values
# Let's find and print out every value greater than zero
values = np.where(alpha_channel != 0)
print ('\n the non-zero values of the alpha channel look like')
print (values)
