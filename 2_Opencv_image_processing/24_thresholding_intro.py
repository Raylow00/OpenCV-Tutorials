import numpy as np
import cv2
from matplotlib import pyplot as plt

def build_sample_image():
    """Builds a sample image with 50x50 regions of different tones of gray"""

    # Define the different tones
    # The end of interval is not included
    # 50, 100, 150, 200, 250
    tones = np.arange(start=50, stop=300, step=50)

    result = np.zeros((50, 50, 3), dtype="uint8")

    for tone in tones:
        img = np.ones((50, 50, 3), dtype="uint8") * tone
        result = np.concatenate((result, img), axis=1)

    return result


def show_img_with_matplotlib(color_img, title, pos):
    # Convert BGR image to RGB
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(7, 1, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

# Create the dimensions of the figure and set title and color:
fig = plt.figure(figsize=(6, 9))
plt.suptitle("Thresholding introduction", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

# Load the image and convert it to grayscale
image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Plot the grayscale images and the histograms
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Image with tones of gray (0, 50, 100, 150, 200, 250)", 1)

# Apply cv2.threshold() with different thresholding values:
# cv2.threshold() takes a grayscale image
# second argument is the threshold value which is used to classify the pixel values
# third argument is the max value which is assigned to pixel values exceeding the threshold
ret1, thresh1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
ret5, thresh5 = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
ret6, thresh6 = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)

# Plot the images
show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "Threshold = 0", 2)
show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "Threshold = 50", 3)
show_img_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "Threshold = 100", 4)
show_img_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "Threshold = 150", 5)
show_img_with_matplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "Threshold = 200", 6)
show_img_with_matplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "Threshold = 250", 7)

plt.show()