"""
OpenCV provides methods to compute the homography matrix - to calculate and find a perspective transformation between two images
1. RANSAC
2. Least Median 
3. PROSAC
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


fig = plt.figure(figsize=(14, 5))
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
best_matches = bf_matches[:40]
print(best_matches)

# Extract the matched keypoints
pts_src = np.float32([keypoints_1[m.queryIdx].pt for m in best_matches]).reshape(-1, 1, 2)
pts_dst = np.float32([keypoints_2[m.trainIdx].pt for m in best_matches]).reshape(-1, 1, 2)

# Find homography matrix
# M is the perspective transformation matrix
# 5.0 is the reprojection threshold where if the reprojection error is > 5.0, the corresponding point pair is considered an outlier
M, mask = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, 5.0)

# Get the corner coordinates of the 'query' image
h, w = image_query.shape[:2]
pts_corners_src = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)

# Perform perspective transform using the previously calculated matrix and the corners of the 'query' image
# to get the corners of the 'detected' object in the 'scene' image
pts_corners_dst = cv2.perspectiveTransform(pts_corners_src, M)

# Draw corners
img_obj = cv2.polylines(image_scene, [np.int32(pts_corners_dst)], True, (0, 0, 255), 10)

# Draw matches
img_matching = cv2.drawMatches(image_query, keypoints_1, img_obj, keypoints_2, best_matches, None, matchColor=(255, 255, 0), singlePointColor=(255, 0, 255), flags=0)

show_img_with_matplotlib(img_obj, "Detected object", 1)
show_img_with_matplotlib(img_matching, "Feature matching", 2)

plt.show()