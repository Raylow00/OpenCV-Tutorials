import numpy as np
import cv2
import matplotlib.pyplot as plt

# This function shows an image using the matplotlib functionality
def show_with_matplotlib(color_img, title, pos):
    # pos - position in the figure plot

    # First convert the BGR image to RGB
    rgb = color_img[:, :, ::-1] # all items in the array, reversed

    """
    Other slice notations worth noting:
    a[-2:]      # last two items in the array
    a[:-2]      # everything except the last two items
    a[1::-1]    # the first two items, reversed
    a[:-3:-1]   # the last two items, reversed
    a[-3::-1]   # everything except the last two items, reversed
    """

    ax = plt.subplot(1, 4, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(12, 6))
plt.suptitle("Sobel operator and cv2.addWeighted()", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')

# Different weights given to the image gives an impression of transparency
# cv2.addWeighted() is commonly used to get the output from Sobel operator
image_filtered = cv2.GaussianBlur(image, (3, 3), 0)

gray_image = cv2.cvtColor(image_filtered, cv2.COLOR_BGR2GRAY)

# Gradient is calculated
# Depth of output is set to CV_165 to avoid overflow
# CV_165 = one channel of 2-byte signed integers (16-bit signed integers)
gradient_x = cv2.Sobel(gray_image, cv2.CV_16S, 1, 0, 3)
gradient_y = cv2.Sobel(gray_image, cv2.CV_16S, 0, 1, 3)

# Conversion to unsigned 8-bit integer (0-255)
abs_gradient_x = cv2.convertScaleAbs(gradient_x)
abs_gradient_y = cv2.convertScaleAbs(gradient_y)

# Combine the two images using the same weight
sobel = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)

show_with_matplotlib(image, "Original", 1)
show_with_matplotlib(cv2.cvtColor(abs_gradient_x, cv2.COLOR_GRAY2BGR), "Gradient x", 2)
show_with_matplotlib(cv2.cvtColor(abs_gradient_y, cv2.COLOR_GRAY2BGR), "Gradient y", 3)
show_with_matplotlib(cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR), "Sobel", 4)

plt.show()

