import cv2
from matplotlib import pyplot as plt

# Adaptive thresholding
# cv2.adaptiveThreshold(src, maxVal, adaptiveMethod, thresholdType, blockSize, c)
# maxVal - the value to set if the condition is satisfied
# blockSize - sets the size of the neighbourhood area used to calculate a threshold value of the pixel, usually 3, 5, 7...
# c - constant subtracted from the means or weighted means
# cv2.ADAPTIVE_THRESH_MEAN_C - the threshold value is calculated as the mean of blockSize x blockSize minus c
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C - the threshold value is calculated as the weighted sum of blockSize x blockSize minus c

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(15, 7))
plt.suptitle("Adaptive thresholding", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("../../Images/bathory_vinyl.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

filtered_gray_image = cv2.bilateralFilter(gray_image, 15, 25, 25)

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Gray image", 1)
show_img_with_matplotlib(cv2.cvtColor(filtered_gray_image, cv2.COLOR_GRAY2BGR), "Gray image", 4)

thres1 = cv2.adaptiveThreshold(filtered_gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thres2 = cv2.adaptiveThreshold(filtered_gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 3)
thres3 = cv2.adaptiveThreshold(filtered_gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thres4 = cv2.adaptiveThreshold(filtered_gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 3)

show_img_with_matplotlib(cv2.cvtColor(thres1, cv2.COLOR_GRAY2BGR), "THRESH_MEAN_C, 11, 2", 2)
show_img_with_matplotlib(cv2.cvtColor(thres2, cv2.COLOR_GRAY2BGR), "THRESH_MEAN_C, 31, 3", 3)
show_img_with_matplotlib(cv2.cvtColor(thres3, cv2.COLOR_GRAY2BGR), "THRESH_GAUSSIAN_C, 11, 2", 5)
show_img_with_matplotlib(cv2.cvtColor(thres4, cv2.COLOR_GRAY2BGR), "THRESH_GAUSSIAN_C, 31, 3", 6)

plt.show()