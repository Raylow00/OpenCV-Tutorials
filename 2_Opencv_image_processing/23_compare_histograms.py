import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

# To get a numerical parameter expressing how well two histograms match each other

image_names = ['gray_image.png', 'gray_blurred.png', 'gray_added_image.png', 'gray_subtracted_image.png']
path = "../../Images"

def load_test_images():
    images = []
    for index_image, name_image in enumerate(image_names):
        image_path = os.path.join(path, name_image)
        images.append(cv2.imread(image_path, 0))

    return images

def show_img_with_matplotlib(color_img, title, pos):
    # Convert from BGR to RGB
    img_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(4, 5, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib_gray(hist, title, pos, color):
    
    ax = plt.subplot(2, 5, pos)
    plt.title(title)
    plt.xlabel('bins')
    plt.ylabel('number of pixels')
    plt.xlim([0, 256])
    plt.plot(hist, color=color)

plt.figure(figsize=(18, 9))
plt.suptitle("Grayscale histogram comparison", fontsize=14, fontweight='bold')

test_images = load_test_images()
hists = []
for image in test_images:
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
    hists.append(hist)

# Returns a numerical value
gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_CORREL)
gray_gray_blurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
gray_added = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CORREL)
gray_sub = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CORREL)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "gray", 1)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "image 1 " + str('CORREL % 6.5f' % gray_gray), 2)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR), "image 2 " + str('CORREL % 6.5f' % gray_gray_blurred), 3)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR), "image 3 " + str('CORREL % 6.5f' % gray_added), 4)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR), "image 4 " + str('CORREL % 6.5f' % gray_sub), 5)

gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_CHISQR)
gray_gray_blurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CHISQR)
gray_added = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_CHISQR)
gray_sub = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_CHISQR)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "gray", 6)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "image 1 " + str('CHISQR % 6.5f' % gray_gray), 7)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR), "image 2 " + str('CHISQR % 6.5f' % gray_gray_blurred), 8)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR), "image 3 " + str('CHISQR % 6.5f' % gray_added), 9)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR), "image 4 " + str('CHISQR % 6.5f' % gray_sub), 10)

gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_INTERSECT)
gray_gray_blurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_INTERSECT)
gray_added = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_INTERSECT)
gray_sub = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_INTERSECT)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "gray", 11)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "image 1 " + str('INTERSECT % 6.5f' % gray_gray), 12)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR), "image 2 " + str('INTERSECT % 6.5f' % gray_gray_blurred), 13)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR), "image 3 " + str('INTERSECT % 6.5f' % gray_added), 14)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR), "image 4 " + str('INTERSECT % 6.5f' % gray_sub), 15)

gray_gray = cv2.compareHist(hists[0], hists[0], cv2.HISTCMP_BHATTACHARYYA)
gray_gray_blurred = cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_BHATTACHARYYA)
gray_added = cv2.compareHist(hists[0], hists[2], cv2.HISTCMP_BHATTACHARYYA)
gray_sub = cv2.compareHist(hists[0], hists[3], cv2.HISTCMP_BHATTACHARYYA)

show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "gray", 16)
show_img_with_matplotlib(cv2.cvtColor(test_images[0], cv2.COLOR_GRAY2BGR), "image 1 " + str('BHATTACHARYYA % 6.5f' % gray_gray), 17)
show_img_with_matplotlib(cv2.cvtColor(test_images[1], cv2.COLOR_GRAY2BGR), "image 2 " + str('BHATTACHARYYA % 6.5f' % gray_gray_blurred), 18)
show_img_with_matplotlib(cv2.cvtColor(test_images[2], cv2.COLOR_GRAY2BGR), "image 3 " + str('BHATTACHARYYA % 6.5f' % gray_added), 19)
show_img_with_matplotlib(cv2.cvtColor(test_images[3], cv2.COLOR_GRAY2BGR), "image 4 " + str('BHATTACHARYYA % 6.5f' % gray_sub), 20)


plt.show()

# Image 1 without any blur or addition and subtraction gives the best metric
# Image 2 gives only good performance as it is a smoothed version
# Image 3 and 4 give poor performance as the histogram is shifted