import time
from lib2to3.pgen2 import driver

import cv2
import pandas as pd
import label as label
import numpy as np
from PIL import Image
from keras import models
import webbrowser
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from selenium import webdriver
import time

from selenium.webdriver.common.devtools.v101 import browser

classNames = []
classFile = 'labels.txt'

# open file
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
print(classNames)
# Load the saved model

model = models.load_model('model_svm_beans.h5')
video = cv2.VideoCapture(0)

data = np.ndarray(shape=(1, 64, 64, 3), dtype=np.float32)

def scroll(predicted_className):
    if predicted_className == 0:
        name = 'Grade 1'

    if predicted_className == 1:
        name = 'Grade 2'

    if predicted_className == 2:
        name = 'Grade 3'

    if predicted_className == 3:
        name = 'Grade 4'

    time.sleep(2)

while True:
    _, frame = video.read()
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((64, 64))
    img_array = np.array(im)

    img_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = img_array

    prediction = model.predict(data)
    predicted_className = prediction.argmax()

    if predicted_className == 0:
        name = 'Grade 1'
    if predicted_className == 1:
        name = 'Grade 2'
    if predicted_className == 2:
        name = 'Grade 3'
    if predicted_className == 3:
        name = 'Grade 4'

    print("predicted: ",predicted_className, " ",name)
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1)

    scroll(predicted_className)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()