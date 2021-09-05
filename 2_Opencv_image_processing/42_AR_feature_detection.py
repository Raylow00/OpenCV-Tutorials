"""
OpenCV provides many techniques and algorithms to detect features in images
1. Harris corner detection
2. Shi-Tomasi corner detection
3. Scale Invariant Feature Transform (SIFT)
4. Speeded-Up Robust Features (SURF)
5. Features from Accelerated Segment Test (FAST)
6. Binary Robust Independent Elementary Features (BRIEF)
7. Oriented FAST and Rotated BRIEF (ORB)
"""

import cv2
import matplotlib.pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')


fig = plt.figure(figsize=(12, 5))
plt.suptitle("ORB Keypoint Detector", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("../../Images/shapes.png")

# Initiate the ORB detector
orb = cv2.ORB_create()

# Detect keypoints using ORB
keypoints = orb.detect(image, None)

# Compute the descriptors of the detected keypoints
keypoints, descriptors = orb.compute(image, keypoints)

# Print one ORB descriptor
print("First extracted descriptor: {}".format(descriptors[0]))

# Draw the detected keypoints
image_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(255, 0, 255), flags=0)

show_img_with_matplotlib(image, "Original image", 1)
show_img_with_matplotlib(image_keypoints, "Detected keypoints", 2)

plt.show()

# Harris Corner Detection
# 1. Take a window
# 2. Take the sum of squared differences (SSD) of the pixel values before and after the shift in window and identify the pixel windows where the SSD is large for shifts in all 8 windows.
# 3. Using the change function that computes the sum of all SSDs, we have to maximize the function for corner detection
# 4. We reach an equation that requires eigenvectors to be solved.
# 5. Solving the eigenvectors, we can obtain the directions for both the largest ant smallest increases in SSD
# 6. R > 0 - corner; R < 0 - edge; R small - flat

# FAST
# 1. Select a pixel p, to be used as an interest point, its intensity be Ip
# 2. Select an appropriate threshold, t
# 3. Consider a circle of 16 pixels around it called the Bresenham circle of radius 3
# 4. Compare the intensity of pixels of the top, bottom, left, right (px 1, 5, 9, 13) 
# 5. If there exists a set of n (=12) contiguous pixels in the circle, brighter than Ip + t, or darker than Ip - t, pixel p is a corner
# 6. If at least 3 of the 4 pixel values, I1, I5, I9 and I13 are not above or below Ip + t, then p is not a corner.
# 7. If at least 3 of the pixels are above or below Ip + t, then check for all 16 pixels and check if 12 contiguous pixels fall in the criterion

# BRIEF
# 1. Starts by smoothing image using a Gaussian kernel to prevent the descriptor from being sensitive to high-frequency noise
# 2. Select a random pair of pixels in a defined neighbourhood around the keypoint, the neighbourhood is called a patch
# 3. The first pixel in the random pair is drawn from a Gaussian distribution centered around the keypoint with a standard deviation.
# 4. The second pixel is centered around the first pixel with a standard deviation divided by 2.
# 5. If the first pixel is brighter than the second, it assigns a value of 1 to the corresponding bit else 0.
# 6. BRIEF repeats this for 128 times for a keypoint with a 128-bit vector