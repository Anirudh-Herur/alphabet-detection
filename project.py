from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import ssl
import os
import cv2
from PIL import Image
import PIL.ImageOps

if(not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('Digit Recognition 2\image.npz')['arr_0']
Y = pd.read_csv('Digit Recognition 2\labels.csv')['labels']

x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=2500, train_size=7500, random_state=42)
x_train_scaled = x_train / 255.0
x_test_scaled = x_test / 255.0
model = LogisticRegression(
    solver='saga', multi_class='multinomial', random_state=42)
model.fit(x_train_scaled, y_train)
y_predict = model.predict(x_test_scaled)
print(accuracy_score(y_test, y_predict))

cam = cv2.VideoCapture(0)
while(True):
    ret, frame = cam.read()
    print(ret)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    upper_left = (int(width/2 - 56), int(height/2 - 56))
    bottom_right = (int(width/2 + 56), int(height/2 + 56))
    cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)
    roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    im_pill = Image.fromarray(roi)
    Image_bw = im_pill.convert('L')
    Image_bw_resize = Image_bw.resize((28, 28), Image.ANTIALIAS)
    Image_bw_resize_inverted = PIL.ImageOps.invert(Image_bw_resize)
    pixel_filter = 20
    min_pixel = np.percentile(Image_bw_resize_inverted, pixel_filter)
    Image_bw_resize_inverted_scaled = np.clip(
        Image_bw_resize_inverted - min_pixel, 0, 255)
    max_pixel = np.max(Image_bw_resize_inverted)
    Image_bw_resize_inverted_scaled = np.asarray(
        Image_bw_resize_inverted_scaled) / max_pixel
    test_sample = np.array([Image_bw_resize_inverted_scaled]).reshape(1, 660)
    y_predict = model.predict(test_sample)
    print(y_predict)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
