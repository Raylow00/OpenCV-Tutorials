import numpy as np
import cv2
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    # Convert from BGR to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_rgb(hist, title, pos, color):
    
    ax = plt.subplot(2, 3, pos)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    
    for (h, c) in zip(hist, color):
        plt.plot(h, color=c)

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    
    ax = plt.subplot(2, 3, pos)
    plt.title(title)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

def hist_color_img(img):
    # Calculates the histogram for a three-channel image
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))   # blue channel
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))   # green channel
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))   # red channel
    
    return histr

def equalize_hist_color(img):
    # Equalize the image splitting the image applying cv2.equalizeHist() to each channel and merging the results

    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channel = cv2.equalizeHist(ch)
        eq_channels.append(eq_channel)

    eq_image = cv2.merge(eq_channels)
    return eq_image

def equalize_hist_color_hsv(img):
    h, s, v = cv2.split(img, cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_v = cv2.equalizeHist(v)
    eq_image = cv2.cvtColor(cv2.merge([h, s, eq_v]), cv2.COLOR_HSV2BGR)
    return eq_image

# Grayscale
""" plt.figure(figsize=(18, 6))
plt.suptitle("Grayscale histogram equalization using equalizeHist()", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist(gray_image, [0], None, [256], [0, 256])
gray_image_eq = cv2.equalizeHist(gray_image)
hist_eq = cv2.calcHist(gray_image_eq, [0], None, [256], [0, 256])

# Plotting
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray image", 1)
show_hist_with_matplotlib_gray(hist, "Histogram grayscale without equalization", 2, 'm')

show_img_with_matplotlib(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "gray image after equalization", 3)
show_hist_with_matplotlib_gray(hist_eq, "Histogram grayscale with equalization", 4, 'm')
 """
# Color
""" plt.figure(figsize=(18, 6))
plt.suptitle("Color histogram equalization using equalizeHist()", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
hist_color = hist_color_img(image)
image_eq = equalize_hist_color(image)
hist_image_eq = hist_color_img(image_eq)

# Plotting
show_img_with_matplotlib(image_eq, "Equalized image", 1)
show_hist_with_matplotlib_rgb(hist_color, "Histogram color without equalization", 2, 'm')
show_hist_with_matplotlib_rgb(hist_image_eq, "Histogram color with equalization", 3, 'm')
 """

# Color - changing color space
plt.figure(figsize=(18, 6))
plt.suptitle("Color histogram equalization using equalizeHist() and using the HSV color space", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
hist_color = hist_color_img(image)
image_eq = equalize_hist_color_hsv(image)
hist_image_eq = hist_color_img(image_eq)

# Plotting
show_img_with_matplotlib(image_eq, "Equalized image", 1)
show_hist_with_matplotlib_rgb(hist_color, "Histogram color without equalization", 2, 'm')
show_hist_with_matplotlib_rgb(hist_image_eq, "Histogram color with equalization", 3, 'm')

plt.show()
