import cv2
import numpy as np
import matplotlib.pyplot as plt

# Different methods for smoothing images

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

    ax = plt.subplot(3, 3, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    #plt.show()

plt.figure(figsize=(16, 9))
plt.suptitle("Smoothing techniques", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')

# We create the kernel for smoothing images
# (10, 10)
kernel_averaging_10_10 = np.ones((10, 10), np.float32) / 100
# But if you know the values, you can put them in directly
kernel_averaging_5_5 = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04],
                                 [0.04, 0.04, 0.04, 0.04, 0.04]])
print("Kernel: {}".format(kernel_averaging_5_5))

#-----------------------------------------------------
# cv2.filter2D() - arbitrary linear filter
smooth_image_f2d_5_5 = cv2.filter2D(image, -1, kernel_averaging_5_5)
smooth_image_f2d_10_10 = cv2.filter2D(image, -1, kernel_averaging_10_10)

#-----------------------------------------------------
# cv2.blur() - uses a normalized box filter
smooth_image_blur = cv2.blur(image, (10, 10))

#-----------------------------------------------------
# When the parameter 'normalize' of cv2.boxFilter() is True
# cv2.filter2D() and cv2.boxFilter() perform the same operation
smooth_image_box_filter = cv2.boxFilter(image, -1, (10, 10), normalize=True)

#-----------------------------------------------------
# Gaussian blur - convolves the image with the specified Gaussian kernel
# Params:
# ksize - kernel size
# sigmaX - standard deviation in the x direction of the gaussian kernel
# sigmaY - standard deviation in the y direction of the gaussian kernel
gaussian_blur = cv2.GaussianBlur(image, (9, 9), 0)

#-----------------------------------------------------
# Median blur - convolves using a median kernel
# Param:
# ksize
median_blur = cv2.medianBlur(image, 9)

#-----------------------------------------------------
# BilateralFilter 
bilateral_filter = cv2.bilateralFilter(image, 5, 10, 10)
bilateral_filter_2 = cv2.bilateralFilter(image, 9, 200, 200)

show_with_matplotlib(image, "Original", 1)
show_with_matplotlib(smooth_image_f2d_5_5, "5x5 kernel", 2)
show_with_matplotlib(smooth_image_f2d_10_10, "10x10 kernel", 3)
show_with_matplotlib(smooth_image_blur, "Blur", 4)
show_with_matplotlib(smooth_image_box_filter, "Box filter", 5)
show_with_matplotlib(gaussian_blur, "Gaussian blur", 6)
show_with_matplotlib(median_blur, "Median blur", 7)
show_with_matplotlib(bilateral_filter, "Bilateral filter", 8)
show_with_matplotlib(bilateral_filter_2, "Bilateral filter 2", 9)

plt.show()