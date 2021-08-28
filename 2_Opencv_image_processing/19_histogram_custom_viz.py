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

def plot_hist(hist_items, color):
    # Plot the histogram of an image
    # For viz purposes, we add some offset
    offset_down = 10
    offset_up = 10
    x_values = np.arange(256).reshape(256, 1)
    canvas = np.ones((300, 256, 3), dtype='uint8') * 255

    for hist_item, col in zip(hist_items, color):
        # Normalize in the range for proper visualization
        cv2.normalize(hist_item, hist_item, 0 + offset_down, 300 - offset_up, cv2.NORM_MINMAX)

        # Round the normalized values of the histogram
        around = np.around(hist_item)

        # Cast the values to int
        hist = np.int32(around)

        # Create the points using the histogram and the x-coordinates
        # x_values = (1, 2, 3)
        # hist = (4, 5, 6)
        # pts = [[1, 4], [2, 5], [3, 6]]
        pts = np.column_stack((x_values, hist))

        # Draw the points
        cv2.polylines(canvas, [pts], False, col, 2)

        # Draw a rectangle
        cv2.rectangle(canvas, (0, 0), (255, 298), (0, 0, 0), 1)

    # Flip the image in the up/down direction
    res = np.flipud(canvas)
    return res

plt.figure(figsize=(16, 10))
plt.suptitle("Custom visualization of histograms", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_color = hist_color_img(image)

gray_plot = plot_hist([hist], [(255, 0, 255)])
color_plot = plot_hist(hist_color, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
show_img_with_matplotlib(image, "image", 4)
show_hist_with_matplotlib_gray(hist, "gray image histogram", 2, "m")
show_hist_with_matplotlib_rgb(hist_color, "color image histogram", 3, ['b', 'g', 'r'])
show_img_with_matplotlib(gray_plot, "Grayscale histogram (custom)", 5)
show_img_with_matplotlib(color_plot, "Color histogram (custom)", 6)

plt.show()
