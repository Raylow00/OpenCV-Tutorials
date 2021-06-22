import cv2
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 9))
plt.suptitle("Colormaps", fontsize=14, fontweight='bold')

image = cv2.imread('../../Images/tzuyu.jpg')
gray = image[:, :, ::-1]
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply color map HSV
img_colormap_hsv = cv2.applyColorMap(gray_img, cv2.COLORMAP_HSV)

cv2.imshow("Original image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Matplotlib maps single channel output, defaulting to "perceptially uniform" colormap
# Need to tell it to use the gray colormap
# plt.imshow(gray_img, cmap='gray')
cv2.imshow("Grayscale image", gray_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow("Image after HSV color map is applied", img_colormap_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()

#######################################################################
#
# OpenCV has some colormaps by default
#
colormaps = ['AUTUMN', 'BONE', 'JET', 'WINTER', 'RAINBOW', 'OCEAN', 'SUMMER', 'SPRING', 
                'COOL', 'HSV', 'HOT', 'PINK', 'PARULA']

plt.figure(figsize=(16, 9))
plt.suptitle("Colormaps in OpenCV", fontsize=14, fontweight='bold')

for idx, cmap in enumerate(colormaps):
    result = cv2.applyColorMap(gray_img, idx)
    cv2.imshow("Using " + str(cmap) + " colormap", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
