from typing import final
import cv2
from matplotlib import pyplot as plt

# Since thresholding produces only 2 colors, the final image can possibly have only 2^3 colors

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12,4))
plt.suptitle("Examples of thresholding", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("../../Images/tzuyu.jpg")

ret1, thres1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

(b, g, r) = cv2.split(image)
ret2, thres2 = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)
ret3, thres3 = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
ret4, thres4 = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
final_image = cv2.merge((thres2, thres3, thres4))

show_img_with_matplotlib(image, "Original image", 1)
show_img_with_matplotlib(thres1, "Thresholding without splitting channels", 2)
show_img_with_matplotlib(final_image, "Thresholding each channel and merge", 3)

plt.show()