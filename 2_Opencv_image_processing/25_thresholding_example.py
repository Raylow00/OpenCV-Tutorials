import cv2
from matplotlib import pyplot as plt

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(9,9))
plt.suptitle("Examples of thresholding", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread("../../Images/bathory_vinyl.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

show_img_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Gray image", 1)

ret1, thres1 = cv2.threshold(gray_image, 60, 255, cv2.THRESH_BINARY)
ret2, thres2 = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)
ret3, thres3 = cv2.threshold(gray_image, 80, 255, cv2.THRESH_BINARY)
ret4, thres4 = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY)
ret5, thres5 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
ret6, thres6 = cv2.threshold(gray_image, 110, 255, cv2.THRESH_BINARY)
ret7, thres7 = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
ret8, thres8 = cv2.threshold(gray_image, 130, 255, cv2.THRESH_BINARY)

show_img_with_matplotlib(cv2.cvtColor(thres1, cv2.COLOR_GRAY2BGR), "Threshold=60", 2)
show_img_with_matplotlib(cv2.cvtColor(thres2, cv2.COLOR_GRAY2BGR), "Threshold=70", 3)
show_img_with_matplotlib(cv2.cvtColor(thres3, cv2.COLOR_GRAY2BGR), "Threshold=80", 4)
show_img_with_matplotlib(cv2.cvtColor(thres4, cv2.COLOR_GRAY2BGR), "Threshold=90", 5)
show_img_with_matplotlib(cv2.cvtColor(thres5, cv2.COLOR_GRAY2BGR), "Threshold=100", 6)
show_img_with_matplotlib(cv2.cvtColor(thres6, cv2.COLOR_GRAY2BGR), "Threshold=110", 7)
show_img_with_matplotlib(cv2.cvtColor(thres7, cv2.COLOR_GRAY2BGR), "Threshold=120", 8)
show_img_with_matplotlib(cv2.cvtColor(thres8, cv2.COLOR_GRAY2BGR), "Threshold=130", 9)

plt.show()