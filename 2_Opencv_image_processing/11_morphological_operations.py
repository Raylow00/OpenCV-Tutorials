import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

image_names = ['test1.png', 'test2.png', 'test3.png']
path = '../../Images'

# The 2 kernel sizes used in this testing
kernel_size_3 = (3, 3)
kernel_size_5 = (5, 5)

# This function shows an image using the matplotlib functionality
def show_with_matplotlib(img, title, pos):
    # pos - position in the figure plot

    # First convert the BGR image to RGB
    #rgb = color_img[:, :, ::-1] # all items in the array, reversed

    """
    Other slice notations worth noting:
    a[-2:]      # last two items in the array
    a[:-2]      # everything except the last two items
    a[1::-1]    # the first two items, reversed
    a[:-3:-1]   # the last two items, reversed
    a[-3::-1]   # everything except the last two items, reversed
    """

    ax = plt.subplot(len(image_names), len(morphological_operations) + 1, pos)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')


def load_test_images():
    test_morph_images = []

    for image_index, image_name in enumerate(image_names):
        image_path = os.path.join(path, image_name)
        print("Image path: {}".format(image_path))

        test_morph_images.append(cv2.imread(image_path))

    return test_morph_images


def show_images(array_img, title, pos):

    for image_index, image in enumerate(array_img):
        show_with_matplotlib(image, title + "_" + str(image_index + 1), pos + image_index * (len(morphological_operations) + 1))


def build_kernel(kernel_type, kernel_size):
    
    if kernel_type == cv2.MORPH_ELLIPSE:
        # we build an elliptical kernel
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    elif kernel_type == cv2.MORPH_CROSS:
        # we build a cross-shape kernel
        return cv2.getStructuringElement(cv2.MORPH_CROSS, kernel_size)

    else:   # cv2.MORPH_RECT
        # we build a rectangular kernel
        return cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)


#######################################################################

# Erode
# Areas of the foreground object will become smaller
# Holes within those areas will get bigger
def erode(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    erosion = cv2.erode(image, kernel, iterations=1)
    return erosion


# Dilate
# Areas of the foreground object will become larger
# Holes within those areas shrink
def dilate(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    dilation = cv2.dilate(image, kernel, iterations=1)
    return dilation


# Closing = dilation + erosion
# Holes shrink, then erode to reduce the effect
def closing(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed


# Opening = erosion + dilation
# Fill holes, thus removing salt-pepper noise
def opening(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    return opened


# Closes then opens the image
def closing_and_opening(image, kernel_type, kernel_size):
    closed = closing(image, kernel_type, kernel_size)
    opened = opening(closed, kernel_type, kernel_size)
    return opened


# Opens then closes the image
def opening_and_closing(image, kernel_type, kernel_size):
    opened = opening(image, kernel_type, kernel_size)
    closed = closing(opened, kernel_type, kernel_size)
    return closed


# Morphological gradient - difference between a dilation and erosion 
# (useful for determining the outline of a particular object in an image)
def morph_gradient(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    morph_gradient_image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    return morph_gradient_image


# Top hat
# Difference between input image and opening of the image
# Reveals bright regions of an image on dark backgrounds
# For example, a white license plate on a black car
def tophat(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    tophat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
    return tophat_image


# Black hat
# Difference between input image and closing of the image
# Reveals dark regions of an image on white backgrounds
# For example, black digits on a white license plate
def blackhat(image, kernel_type, kernel_size):
    kernel = build_kernel(kernel_type, kernel_size)
    blackat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
    return blackat_image


#######################################################################

# List all the morphological operations to perform, in a dictionary, with each value representing the function itself
morphological_operations = {
    'erode': erode,
    'dilate': dilate,
    'closing': closing,
    'opening': opening,
    'gradient': morph_gradient,
    'closing|opening': closing_and_opening,
    'opening|closing': opening_and_closing,
    'tophat': tophat,
    'blackhat': blackhat
}

#######################################################################

# Make a function that applies the morphological operations
def apply_operation(array_img, morphological_operation, kernel_type, kernel_size):
    morphological_op_result = []

    for image_index, image in enumerate(array_img):
        result = morphological_operations[morphological_operation](image, kernel_type, kernel_size)
        morphological_op_result.append(result)

    return morphological_op_result

#######################################################################

# Morphological Operations - MORPH RECT + KSIZE (3, 3)

test_images = load_test_images()

plt.figure(figsize=(16, 9))
plt.suptitle("Morphological Operations - MORPH RECT + KSIZE (3, 3)", fontsize=14, fontweight='bold')

show_images(test_images, "test image", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_operation(test_images, k, cv2.MORPH_RECT, kernel_size_3), k, i+2)

plt.show()

#######################################################################

# Morphological Operations - MORPH RECT + KSIZE (5, 5)

plt.figure(figsize=(16, 9))
plt.suptitle("Morphological Operations - MORPH RECT + KSIZE (5, 5)", fontsize=14, fontweight='bold')

show_images(test_images, "test image", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_operation(test_images, k, cv2.MORPH_RECT, kernel_size_5), k, i+2)

plt.show()

#######################################################################

# Morphological Operations - MORPH CROSS + KSIZE (3, 3)

test_images = load_test_images()

plt.figure(figsize=(16, 9))
plt.suptitle("Morphological Operations - MORPH CROSS + KSIZE (3, 3)", fontsize=14, fontweight='bold')

show_images(test_images, "test image", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_operation(test_images, k, cv2.MORPH_CROSS, kernel_size_3), k, i+2)

plt.show()

#######################################################################

# Morphological Operations - MORPH CROSS + KSIZE (5, 5)

test_images = load_test_images()

plt.figure(figsize=(16, 9))
plt.suptitle("Morphological Operations - MORPH RECT + KSIZE (3, 3)", fontsize=14, fontweight='bold')

show_images(test_images, "test image", 1)

for i, (k, v) in enumerate(morphological_operations.items()):
    show_images(apply_operation(test_images, k, cv2.MORPH_CROSS, kernel_size_5), k, i+2)

plt.show()