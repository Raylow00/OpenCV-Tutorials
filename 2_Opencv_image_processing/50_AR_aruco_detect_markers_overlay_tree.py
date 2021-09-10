import cv2
import os
import pickle
import numpy as np

OVERLAY_SIZE_PER = 1

overlay_img = cv2.imread("../../Images/tree_overlay.png")

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

    img = cv2.drawContours(img, [pts], -1, (255, 255, 0), -3)

    for p in pts:
        cv2.circle(img, (p[0], p[1]), 5, (255, 0, 255), -1)

    return img


# Place overlay image
def draw_augmented_overlay(pts_1, overlay, img):

    # First define how big the overlay image is by defining a square bounding the overlay image
    pts_2 = np.float32([[0, 0], [overlay.shape[1], 0], [overlay.shape[1], overlay.shape[0]], [0, overlay.shape[0]]])

    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], overlay.shape[0]), (255, 255, 0), 10)

    # Transformation matrix
    M = cv2.getPerspectiveTransform(pts_2, pts_1)

    # Transform the overlay image 
    dst_image = cv2.warpPerspective(overlay, M, (img.shape[1], img.shape[0]))

    # Create the mask
    dst_img_gray = cv2.cvtColor(dst_image, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(dst_img_gray, 0, 255, cv2.THRESH_BINARY_INV)

    # Compute bitwise conjunction
    image_masked = cv2.bitwise_and(img, img, mask=mask)

    result = cv2.add(dst_image, image_masked)

    print("Overlaying...")

    return result

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

            # Overlay
            frame = draw_augmented_overlay(projected_desired_pts, overlay_img, frame)

            # Draw the projected points
            draw_points(frame, projected_desired_pts)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
