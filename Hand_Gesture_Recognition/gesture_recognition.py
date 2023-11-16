import os
import cv2
from visualize import *
import time

dir_path = os.path.dirname(os.path.abspath(__file__))

gesture = "gesture_recognizer.task"
model_path = os.path.join(dir_path, gesture)


import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup Detection, Recognition
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Webcam Stream loopcreate_from
import cv2 
cam = cv2.VideoCapture(0)
current_time = 0
prev_time = 0
while True:

    current_time = time.time()
    fps = 1/(current_time-prev_time)
    prev_time = current_time

    _, BGR = cam.read()
    BGR_time = BGR.copy()
    cv2.putText(BGR_time, "FPS: "+str(round(fps,1)), (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)
    cv2.imshow("Original Frame", BGR_time)


    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=RGB) # Convert to mp image
    result = recognizer.recognize(mp_image)  # Detection Result
    try:
        best_guess = result.gestures[0][0]
        annotated = RGB.copy()# Annotate

        cv2.putText(annotated, "Gesture: "+best_guess, (10,50), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255),2)

        cv2.imshow("Annotated", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    except:
        print("No hand")


    if cv2.waitKey(1) == ord("q"):
        break