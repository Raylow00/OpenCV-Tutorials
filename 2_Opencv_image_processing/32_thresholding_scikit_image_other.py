import cv2
from matplotlib import pyplot as plt
from skimage.filters import threshold_otsu, threshold_niblack, threshold_triangle, threshold_sauvola
from skimage import img_as_ubyte

# Otsu thresholding
# Suitable for bimodal images where there are two obvious peaks in its histogram
# Calculates the optimal threshold value that separates both peaks by maximizing the variance between two classes of pixels

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color, t=-1):
    
    ax = plt.subplot(2, 3, pos)
    plt.title(title)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color=color)

fig = plt.figure(figsize=(12, 8))
plt.suptitle("Otsu's binarization algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("../../Images/Polygons.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

## Otsu from scikit image 
# Calculate its histogram only for vizualisation
hist_otsu = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
# Returns the threshold value
thresh_otsu = threshold_otsu(gray_image)
# Builds the binary image - an array of boolean
binary_otsu = gray_image > thresh_otsu
# Convert to uint8 data type
binary_otsu = img_as_ubyte(binary_otsu)


## Niblack's from scikit image
thresh_niblack = threshold_niblack(gray_image, window_size=25, k=0.8)
binary_niblack = gray_image > thresh_niblack
binary_niblack = img_as_ubyte(binary_niblack)


## Sauvola's from scikit image
thresh_sauvola = threshold_sauvola(gray_image, window_size=25)
binary_sauvola = gray_image > thresh_sauvola
binary_sauvola = img_as_ubyte(binary_sauvola)


## Triangle from scikit image
thresh_triangle = threshold_triangle(gray_image)
binary_triangle = gray_image > thresh_triangle
binary_triangle = img_as_ubyte(binary_triangle)


show_img_with_matplotlib(image, "Color image", 1)
show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Gray image", 2)
show_img_with_matplotlib(cv2.cvtColor(binary_otsu, cv2.COLOR_GRAY2BGR), "Thresholding using Scikit Image (Otsu)", 3)
show_img_with_matplotlib(cv2.cvtColor(binary_niblack, cv2.COLOR_GRAY2BGR), "Thresholding using Scikit Image (Niblack)", 4)
show_img_with_matplotlib(cv2.cvtColor(binary_sauvola, cv2.COLOR_GRAY2BGR), "Thresholding using Scikit Image (Sauvola)", 5)
show_img_with_matplotlib(cv2.cvtColor(binary_triangle, cv2.COLOR_GRAY2BGR), "Thresholding using Scikit Image (Triangle)", 6)


plt.show()