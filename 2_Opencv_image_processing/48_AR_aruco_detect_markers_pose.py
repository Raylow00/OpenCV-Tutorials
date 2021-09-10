import cv2
import os
import pickle

# Check for the camera calibration pickle data 
if not os.path.exists('calibration2.pckl'):
    print("You need to calibrate the camera you'll be using. See script 47")
    exit()

else:
    f = open('calibration2.pckl', 'rb')
    cameraMatrix, distCoeffs = pickle.load(f)
    f.close()

    if cameraMatrix is None or distCoeffs is None:
        print("Calibration issue. Remove the calibration file and recalibrate your camera")
        exit()

aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# Create params to be used when detecting markers
params = cv2.aruco.DetectorParameters_create()

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # List of ids and corners
    corners, ids, rejectedImgPts = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=params)

    # Draw detected markers
    frame = cv2.aruco.drawDetectedMarkers(image = frame, corners = corners, ids = ids, borderColor = (0, 255, 0))

    # Draw rejected markers
    frame = cv2.aruco.drawDetectedMarkers(image = frame, corners = rejectedImgPts, borderColor = (0, 0, 255))

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec, tvec, 1)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
