# Comparing different kernels using cv2.filter2D()

# Sobel vs Laplacian
# Sobel operator (df/dx, df/dy) takes the first derivative of the intensity of images - a maximum is characterized as an edge
# Laplacian operator (d2f/dx2, d2f/dy2) takes the second derivative where 0 is the edge, but meaningless locations have 0 too - can be solved by applying filtering

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

    ax = plt.subplot(3, 4, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(12, 6))
plt.suptitle("Comparing different kernels using cv2.filter2D()", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')

# Kernels
kernel_identity = np.array([[0, 0, 0], 
                            [0, 1, 0], 
                            [0, 0, 0]])

# Edge detection kernels
kernel_edge_detection_1 = np.array([[1, 0, -1], 
                                    [0, 0, 0], 
                                    [-1, 0, 1]])

kernel_edge_detection_2 = np.array([[0, 1, 0], 
                                    [1, -4, 1], 
                                    [0, 1, 0]])

kernel_edge_detection_3 = np.array([[-1, -1, -1], 
                                    [-1, 8, -1], 
                                    [-1, -1, -1]])

sobel_x_kernel = np.array([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]])

sobel_y_kernel = np.array([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]])

outline_kernel = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

# Sharpening kernels
kernel_sharpen = np.array([[0, -1, 0], 
                            [-1, 5, -1], 
                            [0, -1, 0]])

kernel_unsharp_masking = -1 / 256 * np.array([[1, 4, 6, 4, 1],
                                                [4, 16, 24, 16, 4],
                                                [6, 24, -476, 24, 6],
                                                [4, 16, 24, 16, 4],
                                                [1, 4, 6, 4, 1]])

# Smoothing kernels
kernel_blur = 1 / 9 * np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])

gaussian_blur = 1 / 16 * np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]])

# Embossing kernel
kernel_emboss = np.array([[1, 0, -1],
                        [-1, 1, 1],
                        [0, 1, 2]])                                                                                                                    

# Apply all kernels
original = cv2.filter2D(image, -1, kernel_identity)
edge_1 = cv2.filter2D(image, -1, kernel_edge_detection_1)
edge_2 = cv2.filter2D(image, -1, kernel_edge_detection_2)
edge_3 = cv2.filter2D(image, -1, kernel_edge_detection_3)
sobel_x = cv2.filter2D(image, -1, sobel_x_kernel)
sobel_y = cv2.filter2D(image, -1, sobel_y_kernel)
outline = cv2.filter2D(image, -1, outline_kernel)
sharpen = cv2.filter2D(image, -1, kernel_sharpen)
unsharp_masking = cv2.filter2D(image, -1, kernel_unsharp_masking)
blur = cv2.filter2D(image, -1, kernel_blur)
gaussian = cv2.filter2D(image, -1, gaussian_blur)
emboss = cv2.filter2D(image, -1, kernel_emboss)

# Show
show_with_matplotlib(original, "Identity kernel", 1)
show_with_matplotlib(edge_1, "Edge detection 1", 2)
show_with_matplotlib(edge_2, "Edge detection 2", 3)
show_with_matplotlib(edge_3, "Edge detection 3", 4)
show_with_matplotlib(sobel_x, "Sobel X", 5)
show_with_matplotlib(sobel_y, "Sobel Y", 6)
show_with_matplotlib(outline, "Outline", 7)
show_with_matplotlib(sharpen, "Sharpen", 8)
show_with_matplotlib(unsharp_masking, "Unsharp masking", 9)
show_with_matplotlib(blur, "Blur", 10)
show_with_matplotlib(gaussian, "Gaussian blur", 11)
show_with_matplotlib(emboss, "Emboss", 12)

plt.show()