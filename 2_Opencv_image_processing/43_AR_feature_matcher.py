"""
Feature matchers provided by OpenCV:
1. Brute force matcher (BF matcher)
2. Fast Library for Approximate Nearest Neighbor (FLANN)
"""

import cv2
import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


fig = plt.figure(figsize=(8, 6))
plt.suptitle("ORB Keypoint Detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image_query = cv2.imread("../../Images/imgR.jpg")
image_scene = cv2.imread("../../Images/imgL.jpg")

# Initiate the ORB detector
orb = cv2.ORB_create()

keypoints_1, desc_1 = orb.detectAndCompute(image_query, None)
keypoints_2, desc_2 = orb.detectAndCompute(image_scene, None)

# BF matcher
# First param: Sets the distance measurement (by default it is cv2.NORM_L2)
# Second param: crossCheck (which is False by default) can be set to True in order to return only consistent pairs in the matching process 
#               (The two features in both sets should match each other)
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
bf_matches = bf_matcher.match(desc_1, desc_2)

# Sort the matches in the order of their distance
bf_matches = sorted(bf_matches, key=lambda x: x.distance)

result = cv2.drawMatches(image_query, keypoints_1, image_scene, keypoints_2, bf_matches[:20], None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

show_img_with_matplotlib(result, "Matches between two images", 1)

plt.show()