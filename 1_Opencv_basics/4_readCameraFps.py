import cv2
import argparse
import time

capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()

    if ret is True:
        processing_start = time.time()
        # ..
        # Processing
        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # ..
        processing_end = time.time()

        # Calculate the difference
        processing_time_frame = processing_end - processing_start

        # FPS = 1 / time_per_frame
        print("FPS: {}".format(1.0 / processing_time_frame))

    else:
        break