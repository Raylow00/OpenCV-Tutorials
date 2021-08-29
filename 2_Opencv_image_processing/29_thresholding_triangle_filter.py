import cv2
from matplotlib import pyplot as plt

# Triangle binarization
# First step: A line is calculated between the maximum of the histogram at bmax on the gray level axis and the lowest value bmin on the gray level axis
# Second step: The distance from the line to the histogram for all the values of b [bmin-bmax] is calculated
# Third step: The level where the distance between the histogram and the line is maximal is chosen as the threshold value

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    
    ax = plt.subplot(3, 2, pos)
    plt.title(title)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)

fig = plt.figure(figsize=(11, 10))
plt.suptitle("Otsu's binarization algorithm with filter", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("../../Images/leaf-noise.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate its histogram only for vizualisation
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

ret1, thresh1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

# Apply Gaussian filter
gray_image_filtered = cv2.GaussianBlur(gray_image, (25, 25), 0)

hist2 = cv2.calcHist([gray_image_filtered], [0], None, [256], [0, 256])

ret2, thresh2 = cv2.threshold(gray_image_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

show_img_with_matplotlib(image, "Image with noise", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Gray image with noise", 2)
show_hist_with_matplotlib_gray(hist, "Gray image histogram", 3, 'm', ret1)
show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "Triangle binarization", 4)
show_hist_with_matplotlib_gray(hist2, "Gray image histogram", 5, 'm', ret2)
show_img_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "Triangle binarization after filter", 6)



plt.show()