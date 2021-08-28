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

def hist_color_img(img):
    # Calculates the histogram for a three-channel image
    histr = []
    histr.append(cv2.calcHist([img], [0], None, [256], [0, 256]))   # blue channel
    histr.append(cv2.calcHist([img], [1], None, [256], [0, 256]))   # green channel
    histr.append(cv2.calcHist([img], [2], None, [256], [0, 256]))   # red channel
    
    return histr


plt.figure(figsize=(15, 6))
plt.suptitle("Color histograms", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')

hist_color = hist_color_img(image)

# Plotting
show_img_with_matplotlib(image, "color image", 1)
show_hist_with_matplotlib_rgb(hist_color, "color image histogram", 4, ['b', 'g', 'r'])

# Add 35 to every pixel so the picture looks lighter and calculate histogram
M = np.ones(image.shape, dtype='uint8') * 35
added_image = cv2.add(image, M)
hist_added_image = hist_color_img(added_image)

show_img_with_matplotlib(added_image, "lighter image", 2)
show_hist_with_matplotlib_rgb(hist_added_image, "color histogram of added image", 5, ['b', 'g', 'r'])

# Subtract 35 from every pixel so the picture looks darker and calculate histogram
M = np.ones(image.shape, dtype='uint8') * 35
subtracted_image = cv2.subtract(image, M)
hist_subtracted_image = hist_color_img(subtracted_image)

show_img_with_matplotlib(subtracted_image, "darker image", 3)
show_hist_with_matplotlib_rgb(hist_subtracted_image, "color histogram of subtracted image", 6, ['b', 'g', 'r'])

plt.show()