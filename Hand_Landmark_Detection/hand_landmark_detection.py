import os
import cv2
from visualize import *
import time

dir_path = os.path.dirname(os.path.abspath(__file__))

hand = "hand_landmarker.task"
model_path = os.path.join(dir_path, hand)

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=10)
detector = vision.HandLandmarker.create_from_options(options)


cam = cv2.VideoCapture(0)
prev_time = 0
current_time = 0
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
    result = detector.detect(mp_image)  # Detection Result


    annotated = draw_landmarks_on_image(RGB, result) # Annotate
    cv2.imshow("Annotated", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    
    if cv2.waitKey(1) == ord("q"):
        break    