import cv2
from matplotlib import pyplot as plt

# Otsu thresholding
# Suitable for bimodal images where there are two obvious peaks in its histogram
# Calculates the optimal threshold value that separates both peaks by maximizing the variance between two classes of pixels

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 2, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    
    ax = plt.subplot(2, 2, pos)
    plt.title(title)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)

fig = plt.figure(figsize=(15, 7))
plt.suptitle("Otsu's binarization algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("../../Images/Polygons.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Calculate its histogram only for vizualisation
hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

ret1, thresh1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_TRUNC + cv2.THRESH_OTSU)

show_img_with_matplotlib(image, "Color image", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Gray image", 2)
show_hist_with_matplotlib_gray(hist, "Gray image histogram", 3, 'm', ret1)
show_img_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "Otsu's binarization", 4)


plt.show()