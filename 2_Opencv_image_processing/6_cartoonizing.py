# Cartoonizing images

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

    ax = plt.subplot(2, 4, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

# Sketches the image applying a Laplacian operator to detect the edges
def sketch_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply median filter
    img_median_blur = cv2.medianBlur(img_gray, 5)

    # Detect edges using Laplacian
    # returns an image with a BLACK background with WHITE edges
    # use cv2.threshold() to invert it to BLACK lines on a WHITE background
    edges = cv2.Laplacian(img_median_blur, cv2.CV_8U, ksize=5)

    # Threshold the edges
    # cv2.THRESH_BINARY - objects appear BLACK on a WHITE background
    # cv2.THRESH_BINARY_INV - objects appear WHITE on a BLACK background
    # When logically OR'ed with Otsu thresholding method, it will calculate the threshold value T automatically based on the histogram with 2 peak values
    ret, thresholded = cv2.threshold(edges, 70, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return thresholded

# Cartoonizes the image by applying cv2.bilateralFilter()
def cartoonize_image(img, gray_mode=False):
    thresholded = sketch_image(img)

    # Apply bilateral filter with big numbers to get the cartoonized effect
    filtered = cv2.bilateralFilter(img, 10, 250, 250)

    # Perform 'bitwise_and' with the thresholded image as mask in order to set these values to the output
    cartoonized = cv2.bitwise_and(filtered, filtered, mask=thresholded)

    if gray_mode:
        return cv2.cvtColor(cartoonized, cv2.COLOR_BGR2GRAY)

    return cartoonized

plt.figure(figsize=(14, 6))
plt.suptitle("Cartoonizing images", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')

custom_sketch_image = sketch_image(image)
custom_cartoonized_image = cartoonize_image(image)
#custom_cartoonized_image_gray = cartoonize_image(image, True)

# OpenCV functions that can get a similar output
sketch_gray, sketch_color = cv2.pencilSketch(image, sigma_s=30, sigma_r=0.1, shade_factor=0.1)
stylized_image = cv2.stylization(image, sigma_s=60, sigma_r=0.07)

show_with_matplotlib(image, 'Image', 1)
show_with_matplotlib(cv2.cvtColor(custom_sketch_image, cv2.COLOR_GRAY2BGR), "Custom sketch", 2)
show_with_matplotlib(cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR), "Sketch gray with pencilSketch", 3)
show_with_matplotlib(sketch_color, "Sketch color with pencilSketch", 4)
show_with_matplotlib(stylized_image, "stylization", 5)
show_with_matplotlib(custom_cartoonized_image, "Custom cartoonized", 6)
#show_with_matplotlib(cv2.cvtColor(custom_cartoonized_image_gray, cv2.COLOR_GRAY2BGR), "Custom cartoonized gray", 7)

plt.show()