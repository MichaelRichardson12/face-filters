from keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt

model = load_model('face_feature_model.h5')
# model.summary()

image = cv2.imread('images/obamas4.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

sunglasses = cv2.imread("images/sunglasses_4.png", cv2.IMREAD_UNCHANGED)
# print ('The sunglasses image has shape: ' + str(np.shape(sunglasses)))

# Extract the pre-trained face detector from an xml file
face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray, 2, 3)

print('Number of faces detected:', len(faces))

image_with_detections = np.copy(image)
size = 60
kernel = np.ones((size, size),np.float32)/(size*size)
# Get the bounding box for each detected face
for (x,y,w,h) in faces:
    # Get the face bounding box
    resized = cv2.resize(gray[y:y+h, x:x+w] ,(96, 96))
    resized_color = cv2.resize(image[y:y+h, x:x+w], (96, 96))
    X = np.vstack(resized) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)
    X = X.reshape(-1, 96, 96, 1) # return each images as 96 x 96 x 1
    keypoints = model.predict(X)[0]
    keypoints = keypoints * 48 + 48 # undo the normalization
    keyx = keypoints[0::2]
    keyy = keypoints[1::2]

    width = abs(keyx[9] - keyx[7])
    scaled_width = width * 1.2
    w_r = scaled_width / np.shape(sunglasses)[1]
    height = int(np.shape(sunglasses)[0] * w_r)
    scaled_width = int(scaled_width)

    new_glasses = cv2.resize(sunglasses, (scaled_width, height))

    y_offset = int(keyy[9] - (scaled_width - width)/2)
    x_offset = int(keyx[9] - (scaled_width - width)/2)
    y1, y2 = y_offset, y_offset + new_glasses.shape[0]
    x1, x2 = x_offset, x_offset + new_glasses.shape[1]

    alpha_g = new_glasses[:, :, 3] / 255.0
    alpha = 1.0 - alpha_g

    for c in range(0, 3):
        resized_color[y1:y2, x1:x2, c] = (alpha_g * new_glasses[:, :, c] + alpha * resized_color[y1:y2, x1:x2, c])

    image[y:y+h, x:x+w] = cv2.resize(resized_color, (h, w))


# Display the image with the detections
fig = plt.figure(4, figsize = (8,8))
ax1 = fig.add_subplot(111)
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_title('Image with Face Filter')
ax1.imshow(image)
fig.savefig('generated/fit_filter.png')
