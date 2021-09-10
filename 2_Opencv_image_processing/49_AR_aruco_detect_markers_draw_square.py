import cv2
import os
import pickle
import numpy as np

OVERLAY_SIZE_PER = 1

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


# Draw points
def draw_points(img, pts):
    pts = np.int32(pts).reshape(-1, 2)
    print(pts)

    img = cv2.drawContours(img, [pts], -1, (255, 255, 0), -3)

    for p in pts:
        cv2.circle(img, (p[0], p[1]), 5, (255, 0, 255), -1)

    return img



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
    #frame = cv2.aruco.drawDetectedMarkers(image = frame, corners = rejectedImgPts, borderColor = (0, 0, 255))

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 1, cameraMatrix, distCoeffs)

        for rvec, tvec in zip(rvecs, tvecs):

            # The marker coordinate system is centered on the center of the marker
            # The coordinates of the four coordinates are:
            # 1: (-markerLength/2, markerLength/2, 0)
            # 2: (markerLength/2, markerLength/2, 0)
            # 3: (markerLength/2, -markerLength/2, 0)
            # 4: (-markerLength/2, -markerLength/2, 0)
            desired_pts = np.float32(
                [[-1/2, 1/2, 0], [1/2, 1/2, 0], [1/2, -1/2, 0], [-1/2, -1/2, 0]]
            ) * OVERLAY_SIZE_PER

            # Project the points
            projected_desired_pts, jac = cv2.projectPoints(desired_pts, rvecs, tvecs, cameraMatrix, distCoeffs)

            # Draw the projected points
            draw_points(frame, projected_desired_pts)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
