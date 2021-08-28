import numpy as np
import cv2
import matplotlib.pyplot as plt

def build_image():
    # Builds a sample image with 50x50 regions of different tones of gray

    tones = np.arange(start=60, stop=240, step=30)  # excluding stop, until 210
    result = np.ones((50, 50, 3), dtype="uint8") * 30

    for tone in tones:
        img = np.ones((50, 50, 3), dtype="uint8") * tone
        result = np.concatenate((result, img), axis=1)

    return result

def build_image_2():
    img = np.fliplr(build_image())
    return img

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    ax = plt.subplot(2, 2, pos)
    plt.title(title)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    plt.plot(hist, color=color)


plt.figure(figsize=(14, 10))
plt.suptitle("Introduction to Grayscale histograms", fontsize=14, fontweight='bold')

image = build_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image_2 = build_image_2()
gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

# Calculate histogram using the cv2.calcHist() function
# cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])
# Arguments:
# 1. List of images to process
# 2. Indexes of the channels to be used to calculate the histogram
# 3. Mask to compute the histogram of a specific region of the image defined by the mask
# 4. A list containing the number of bins for each channel
# 5. The range of possible pixel values
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_2 = cv2.calcHist([gray_image_2], [0], None, [256], [0, 256])

cv2.imshow("Image with 50x50 regions of different tones of gray", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

show_hist_with_matplotlib_gray(hist, 'grayscale histogram', 2, 'm')

cv2.imshow("Image with 50x50 regions of different tones of gray - flipped", image_2)
cv2.waitKey(0)
cv2.destroyAllWindows()

show_hist_with_matplotlib_gray(hist_2, 'grayscale histogram', 4, 'm')
plt.show()

# Histogram Terminology
# bins: the number of pixels (frequency) for every tonal value. OpenCV uses histSize to refer to bins. Common values are 8, 16, 32, 64, 128, 256
# range: range of intensity values we want to measure, usually [0, 255], corresponding to all tonal values