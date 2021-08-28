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

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    
    ax = plt.subplot(2, 3, pos)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(15, 6))
plt.suptitle("Grayscale histogram", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

# Plotting
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_hist_with_matplotlib_gray(hist, "grayscale histogram", 4, "m") # 'm' is the magenta color

# Add 35 to every pixel so the picture looks lighter and calculate histogram
M = np.ones(gray_image.shape, dtype='uint8') * 35
added_image = cv2.add(gray_image, M)
hist_added_image = cv2.calcHist([added_image], [0], None, [256], [0, 256])

show_img_with_matplotlib(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "added", 2)
show_hist_with_matplotlib_gray(hist_added_image, "grayscale histogram of added image", 5, "m")

# Subtract 35 from every pixel so the picture looks darker and calculate histogram
M = np.ones(gray_image.shape, dtype='uint8') * 35
subtracted_image = cv2.subtract(gray_image, M)
hist_subtracted_image = cv2.calcHist([subtracted_image], [0], None, [256], [0, 256])

show_img_with_matplotlib(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "subtracted", 3)
show_hist_with_matplotlib_gray(hist_subtracted_image, "grayscale histogram of subtracted image", 6, "m")

plt.show()