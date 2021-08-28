import numpy as np
import cv2
from matplotlib import pyplot as plt
from timeit import default_timer as timer

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


plt.figure(figsize=(18, 6))
plt.suptitle("Comparing histograms using OpenCV, Numpy, and Matplotlib", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Numpy
start = timer()
hist_np, bins_np = np.histogram(gray_image.ravel(), 256, [0, 256])
end = timer()
exec_time_np_hist = (end-start) * 1000

# OpenCV
start = timer()
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
end = timer()
exec_time_calc_hist = (end-start) * 1000

# Matplotlib
start = timer()
(n, bins, patches) = plt.hist(gray_image.ravel(), 256, [0, 256])
end = timer()
exec_time_plt_hist = (end-start) * 1000

# Plotting
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray image", 1)
show_hist_with_matplotlib_gray(hist_np, "Histogram (Numpy)" + str('% 6.2f ms' % exec_time_np_hist), 2, 'm')
show_hist_with_matplotlib_gray(hist, "Histogram (OpenCV)" + str('% 6.2f ms' % exec_time_calc_hist), 3, 'm')
show_hist_with_matplotlib_gray(n, "Histogram (Matplotlib)" + str('% 6.2f ms' % exec_time_plt_hist), 4, 'm')

plt.show()
