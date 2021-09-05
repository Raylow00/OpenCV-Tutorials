import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


fig = plt.figure(figsize=(12, 5))
plt.suptitle("ORB Keypoint Detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Aruco has some predefined dictionaries
# We are going to create a dictionary which is composed by 250 markers
# Each marker will be of 7x7 bits
aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

# Now we can draw a marker using cv2.aruco.drawMarker()
# It returns the marker image ready to be printed
# First param: dictionary object - aruco_dictionary
# Second param: marker id, which ranges between 0 and 249 (where our dict has 250 markers)
# Third param: size of the image to be drawn, in this case 600x600 px
# Forth param: number of bits in marker borders
aruco_marker_1 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=1)
aruco_marker_2 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=2)
aruco_marker_3 = cv2.aruco.drawMarker(dictionary=aruco_dictionary, id=2, sidePixels=600, borderBits=3)

cv2.imwrite("../../Images/marker_DICT_7X7_250_600_1.png", aruco_marker_1)
cv2.imwrite("../../Images/marker_DICT_7X7_250_600_2.png", aruco_marker_2)
cv2.imwrite("../../Images/marker_DICT_7X7_250_600_3.png", aruco_marker_3)

show_img_with_matplotlib(cv2.cvtColor(aruco_marker_1, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_1", 1)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_2, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_2", 2)
show_img_with_matplotlib(cv2.cvtColor(aruco_marker_3, cv2.COLOR_GRAY2BGR), "marker_DICT_7X7_250_600_3", 3)

plt.show()