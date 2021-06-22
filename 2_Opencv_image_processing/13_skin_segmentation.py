import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from numpy.lib.npyio import load

image_names = ['test1.jpg', 'test2.jpg', 'test3.jpg', 'test4.jpg', 'test5.jpg', 'test6.jpg']
path = '../../Images/skin_test_imgs'

def load_images():
    skin_images = []
    for image_index, image_name in enumerate(image_names):
        image_path = os.path.join(path, image_name)
        skin_images.append(cv2.imread(image_path))

    return skin_images


# Lower and upper boundaries
lower_hsv = np.array([0, 48, 80], dtype='uint8')
upper_hsv = np.array([20, 255, 255], dtype='uint8')

lower_hsv_2 = np.array([0, 50, 0], dtype='uint8')
upper_hsv_2 = np.array([120, 150, 255], dtype='uint8')


# Values taken from publication 'Face Segmentation Using Skin-Color Map in Videophone Applications'
lower_ycrcb = np.array([0, 133, 77], dtype='uint8')
upper_ycrcb = np.array([255, 173, 127], dtype='uint8')


# For HSV and YCrCb color spaces
def skin_detector(img, color_space, lower_boundary, upper_boundary):

    if color_space == "HSV":
        cspace = cv2.COLOR_BGR2HSV
    else:
        cspace = cv2.COLOR_BGR2YCrCb

    image = cv2.cvtColor(img, cspace)
    skin_region = cv2.inRange(image, lower_boundary, upper_boundary)
    
    return skin_region


# For BGR color space
def bgr_skin(b, g, r):
    e1 = bool((r > 95) and (g > 40) and (b > 20) and ((max(r, max(g, b)) - min(r, min(g, b))) > 15) and (abs(int(r) - int(g)) > 15) and (r > g) and (r > b))
    e2 = bool((r > 220) and (g > 210) and (b > 170) and (abs(int(r) - int(g)) <= 15) and (r > b) and (g > b))

    return e1 or e2


def skin_detector_bgr(bgr_image):
    h = bgr_image.shape[0]
    w = bgr_image.shape[1]

    res = np.zeros((h, w, 1), dtype='uint8')

    # Only 'skin pixels' will be set to white 255 in the res image
    for y in range(0, h):
        for x in range(0, w):
            (b, g, r) = bgr_image[y, x]
            if bgr_skin(b, g, r):
                res[y, x] = 255

    return res

#######################################################################
#
# Execution
# Segmentation in BGR color space shows more accurate segmentation
#
test_images = load_images()

for img in test_images:
    result_image = skin_detector(img, "HSV", lower_hsv, upper_hsv)
    plt.imshow(result_image)
    plt.title("Skin segmentation using HSV color space")
    plt.show()

for img in test_images:
    result_image = skin_detector(img, "YCrCb", lower_ycrcb, upper_ycrcb)
    plt.imshow(result_image)
    plt.title("Skin segmentation using YCrCb color space")
    plt.show()

for img in test_images:
    result_image = skin_detector_bgr(img)
    plt.imshow(result_image)
    plt.title("Skin segmentation using BGR color space")
    plt.show()
