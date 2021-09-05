import cv2
import matplotlib.pyplot as plt
import numpy as np

aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# Create the parameters object
parameters = cv2.aruco.DetectorParameters_create()

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.aruco.detectMarkers()
    # First param: image where the markers are going to be detected
    # Second param: dictionary object
    # Third param: Establishes all the parameters that can be customized during the detection process
    # Returns the list of corners and identifiers of the detected markers and list of rejected candidates
    corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, aruco_dictionary, parameters=parameters)

    # Draw detected markers
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=corners, ids=ids, borderColor=(0, 255, 0))

    # Draw rejected markers
    frame = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()