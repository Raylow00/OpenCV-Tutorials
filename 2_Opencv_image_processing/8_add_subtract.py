import numpy as np
import cv2
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

    ax = plt.subplot(2, 3, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(12, 6))
plt.suptitle("Arithmetic with images", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')

# Add 60 to every pixel in the image, result will look lighter
M = np.ones(image.shape, dtype='uint8') * 60
added_image = cv2.add(image, M)

# Subtract 60, result will look darker
subtracted_image = cv2.subtract(image, M)

# Similarly, we can build a scalar and add/subtract it
scalar = np.ones((1, 3), dtype='float') * 110
added_image_2 = cv2.add(image, scalar)
subtracted_image_2 = cv2.subtract(image, scalar)

show_with_matplotlib(image, "Original", 1)
show_with_matplotlib(added_image, "Added 60 (image + image)", 2)
show_with_matplotlib(subtracted_image, "Subtracted 60 (image - image)", 3)
show_with_matplotlib(added_image_2, "Added 110 (image + scalar)", 4)
show_with_matplotlib(subtracted_image_2, "Subtracted 110 (image - scalar)", 5)

plt.show()