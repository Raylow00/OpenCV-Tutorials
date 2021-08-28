import numpy as np
import cv2
from matplotlib import pyplot as plt

# Contrast Limited Adaptive Histogram Equalization

def show_img_with_matplotlib(color_img, title, pos):
    # Convert from BGR to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 5, pos)
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

def equalize_clahe_color(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    channels = cv2.split(img)
    eq_channels = []
    for ch in channels:
        eq_channels.append(cla.apply(ch))
    eq_image = cv2.merge(eq_channels)
    return eq_image

def equalize_clahe_color_hsv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    eq_v = cla.apply(v)
    eq_image = cv2.cvtColor(cv2.merge([h, s, eq_v]), cv2.COLOR_HSV2BGR)
    return eq_image

def equalize_clahe_color_lab(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    L, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    eq_L = cla.apply(L)
    eq_image = cv2.cvtColor(cv2.merge([eq_L, a, b]), cv2.COLOR_Lab2BGR)
    return eq_image

def equalize_clahe_color_yuv(img):
    cla = cv2.createCLAHE(clipLimit=4.0)
    y, u, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
    eq_y = cla.apply(y)
    eq_image = cv2.cvtColor(cv2.merge([eq_y, u, v]), cv2.COLOR_YUV2BGR)
    return eq_image


plt.figure(figsize=(18, 14))
plt.suptitle("Histogram equalization using CLAHE", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0)
gray_image_clahe = clahe.apply(gray_image)

clahe.setClipLimit(5.0)
gray_image_clahe_2 = clahe.apply(gray_image)

clahe.setClipLimit(10.0)
gray_image_clahe_3 = clahe.apply(gray_image)

clahe.setClipLimit(20.0)
gray_image_clahe_4 = clahe.apply(gray_image)

image_clahe_color = equalize_clahe_color(image)
image_clahe_color_hsv = equalize_clahe_color_hsv(image)
image_clahe_color_lab = equalize_clahe_color_lab(image)
image_clahe_color_yuv = equalize_clahe_color_yuv(image)

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe, cv2.COLOR_GRAY2BGR), "gray clahe", 2)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_2, cv2.COLOR_GRAY2BGR), "gray clahe 2", 3)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_3, cv2.COLOR_GRAY2BGR), "gray clahe 3", 4)
show_img_with_matplotlib(cv2.cvtColor(gray_image_clahe_4, cv2.COLOR_GRAY2BGR), "gray clahe 4", 5)

show_img_with_matplotlib(image, "color image", 6)
show_img_with_matplotlib(image_clahe_color, "color image clahe", 7)
show_img_with_matplotlib(image_clahe_color_hsv, "color image clahe hsv", 8)
show_img_with_matplotlib(image_clahe_color_lab, "color image clahe lab", 9)
show_img_with_matplotlib(image_clahe_color_yuv, "color image clahe yuv", 10)

plt.show()