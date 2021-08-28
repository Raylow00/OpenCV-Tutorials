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


# Adding a mask
mask = np.zeros(gray_image.shape[:2], dtype='uint8')
mask[30:190, 30:190] = 255
hist_mask = cv2.calcHist([gray_image], [0], mask, [256], [0, 256])

# We will show the masked image for visualization and the grayscale masked histogram
masked_img = cv2.bitwise_and(gray_image, gray_image, mask=mask)
show_img_with_matplotlib(cv2.cvtColor(masked_img, cv2.COLOR_GRAY2BGR), "masked image", 3)
show_hist_with_matplotlib_gray(hist_mask, "masked histogram", 4, "r")

plt.show()