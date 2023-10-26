import os
import cv2
from visualize import *
import time

dir_path = os.path.dirname(os.path.abspath(__file__))

lite = "pose_landmarker_lite.task"
heavy = "pose_landmarker_heavy.task"
full = "pose_landmarker_full.task"
model_path = os.path.join(dir_path, full)


import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup Detection Object
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

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
    result = detector.detect(mp_image)  # Detection Result


    annotated = draw_landmarks_on_image(RGB, result) # Annotate
    cv2.imshow("Annotated", cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    


    if cv2.waitKey(1) == ord("q"):
        break