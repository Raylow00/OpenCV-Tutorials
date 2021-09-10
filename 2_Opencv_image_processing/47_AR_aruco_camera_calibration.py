import time
import cv2
import numpy as np
import pickle

# cv2.aruco.calibrateCameraCharuco()
# @Brief
# calibrates a camera using a set of corners from several views extracted from a board
# when the process is finished, the function returns a camera matrix (a 3x3 floating point camera matrix)
# and a vector containing distortion coefficients
# The 3x3 matrix encodes both the focal distances and the camera center coordinates(called intrinsic parameters)
# The distortion coefficients model the distortion produced by the camera

# HOW TO PERFORM CAMERA CALIBRATION:
# 1. Prepare a piece of paper containing the charuco board, the size does not matter
# 2. Place it where the camera can clearly see
# 3. The camera matrix will be returned
# 4. Estimate pose (in the next script)

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# Params:
# squareX - the number of squares in the x direction
# squareY - the number of squares in the y direction
# squareLength - the chessboard square side length(m)
# markerLength - marker side length(m)
# dictionary - first marks in the dictionary to use in order to create the markers inside the board
board = cv2.aruco.CharucoBoard_create(3, 3, 0.025, 0.0125, dictionary)

# Create board image to be used in calibration process
image_board = board.draw((200*3, 200*3))

# Write calibration board image
cv2.imwrite("../../Images/charuco.png", image_board)

cap = cv2.VideoCapture(0)

all_corners = []
all_ids = []
counter = 0

for i in range(300):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    res = cv2.aruco.detectMarkers(gray, dictionary)

    if len(res[0]) > 0:
        res2 = cv2.aruco.interpolateCornersCharuco(res[0], res[1], gray, board)

        if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and counter % 3 == 0:
            all_corners.append(res2[1])
            all_ids.append(res2[2])

        cv2.aruco.drawDetectedMarkers(gray, res[0], res[1])

    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    counter += 1

# Calibration can fail for many reasons
try:
    # Params:
    # charucoCorners (here all_corners) - a vector containing the detected charuco corners
    # charucoIds (here all_ids) - the list of identifiers
    # board - board layout
    # imageSize - input image size
    # rvecs - a vector of rotation vectors estimated for each board view
    # tvecs - a vector of translation vectors estimated for each pattern view
    # cameraMatrix - camera matrix
    # distCoeffs - distortion coefficients
    cal = cv2.aruco.calibrateCameraCharuco(all_corners, all_ids, board, gray.shape, None, None)
except:
    cap.release()
    print("Calibration failed...")

# Calibration result
retval, cameraMatrix, distCoeffs, rvecs, tvecs = cal

# Save the camera parameters
f = open('calibration2.pckl', 'wb')
pickle.dump((cameraMatrix, distCoeffs), f)
f.close()

cap.release()
cv2.destroyAllWindows()
