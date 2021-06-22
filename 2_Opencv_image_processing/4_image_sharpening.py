# Unsharp masking
# where an unsharped/smoothed version of an image is subtracted from the original image

import cv2
import numpy as np
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

    ax = plt.subplot(6, 1, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

def unsharped_filter(img):
    smoothed = cv2.GaussianBlur(img, (9, 9), 10)
    return cv2.addWeighted(img, 1.5, smoothed, -0.5, 0)

# Create the dimensions of the figure and set title
plt.figure(figsize=(9, 16))
plt.title("Sharpening images", fontsize=14, fontweight='bold')

# Load the image
image = cv2.imread('../../Images/tzuyu.jpg')

# Create kernels for sharpening images and compare with this method of unsharp masking
kernel_1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
kernel_2 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel_3 = np.array([[1, 1, 1], [1, 7, 1], [1, 1, 1]])
kernel_4 = np.array([[-1, -1, -1, -1, -1], [-1, 2, 2, 2, -1], [-1, 2, 8, 2, -1], [-1, 2, 2, 2, -1], [-1, -1, -1, -1, -1]]) / 8.0

# Apply all the kernels
sharpen_image_1 = cv2.filter2D(image, -1, kernel_1)
sharpen_image_2 = cv2.filter2D(image, -1, kernel_2)
sharpen_image_3 = cv2.filter2D(image, -1, kernel_3)
sharpen_image_4 = cv2.filter2D(image, -1, kernel_4)

sharpen_image_5 = unsharped_filter(image)

show_with_matplotlib(image, "Original", 1)
show_with_matplotlib(sharpen_image_1, "Sharp 1", 2)
show_with_matplotlib(sharpen_image_2, "Sharp 2", 3)
show_with_matplotlib(sharpen_image_3, "Sharp 3", 4)
show_with_matplotlib(sharpen_image_4, "Sharp 4", 5)
show_with_matplotlib(sharpen_image_5, "Sharp 5", 6)

plt.show()