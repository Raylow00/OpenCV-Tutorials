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

    ax = plt.subplot(6, 1, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(9, 16))
plt.suptitle("Bitwise Operations", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/181.jpg')
image2 = cv2.imread('../../Images/Tesla.jpg')

# Bitwise AND
bitwise_and = cv2.bitwise_and(image, image2)

# Bitwise OR
bitwise_or = cv2.bitwise_or(image, image2)

# Bitwise XOR
bitwise_xor = cv2.bitwise_xor(image, image2)

# Bitwise NOT
bitwise_not = cv2.bitwise_not(image, image2)

show_with_matplotlib(image, "Original Image 1", 1)
show_with_matplotlib(image2, "Original Image 2", 2)
show_with_matplotlib(bitwise_and, "Bitwise AND", 3)
show_with_matplotlib(bitwise_or, "Bitwise OR", 4)
show_with_matplotlib(bitwise_xor, "Bitwise XOR", 5)
show_with_matplotlib(bitwise_not, "Bitwise NOT", 6)

plt.show()
