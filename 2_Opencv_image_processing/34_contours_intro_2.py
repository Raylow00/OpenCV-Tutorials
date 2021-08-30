import numpy as np
import cv2
from matplotlib import pyplot as plt

# RETR_EXTERNAL: outputs only the external contours
# RETR_LIST: outputs all the contours without any hierarchical relationship
# RETR_TREE: outputs all the contours by establishing a hierachical relationship

def get_one_contour():
    cnts = [np.array([[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
         [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]
    return cnts

def array_to_tuple(arr):
    return tuple(arr.reshape(1, -1)[0])

def draw_contour_points(img, cnts, color):
    for cnt in cnts:
        squeeze = np.squeeze(cnt)

        for p in squeeze:
            p = array_to_tuple(p)
            cv2.circle(img, p, 10, color, -1)

    return img

def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)

def show_img_with_matplotlib(color_img, title, pos):
    img_RGB = color_img[:, :, ::-1]
    ax = plt.subplot(2, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

def build_sample_image():
    img = np.ones((500, 500, 3), dtype='uint8') * 70
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)

    return img

def build_sample_image_2():
    img = np.ones((500, 500, 3), dtype='uint8')
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 255), -1)
    cv2.rectangle(img, (150, 150), (250, 250), (70, 70, 70), -1)
    cv2.circle(img, (400, 400), 100, (255, 255, 0), -1)
    cv2.circle(img, (400, 400), 50, (70, 70, 70), -1)

    return img

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Contours introduction", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray_image, 70, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours3, hierarchy3 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
contours4, hierarchy4 = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)

print("detected contours (RETR EXTERNAL): '{}'".format(len(contours)))
print("detected contours (RETR_LIST): '{}'".format(len(contours2)))

image_contours = image.copy()
image_contours_2 = image.copy()
image_contours_3 = image.copy()
image_contours_4 = image.copy()

draw_contour_outline(image_contours, contours, (0, 0, 255), 5)
draw_contour_outline(image_contours_2, contours2, (255, 0, 0), 5)
draw_contour_outline(image_contours_3, contours3, (255, 0, 0), 5)
draw_contour_outline(image_contours_4, contours4, (255, 0, 0), 5)

draw_contour_points(image_contours, contours, (255, 255, 255))
draw_contour_points(image_contours_2, contours2, (255, 255, 255))
draw_contour_points(image_contours_3, contours3, (255, 255, 255))
draw_contour_points(image_contours_4, contours4, (255, 255, 255))

show_img_with_matplotlib(image, "Original image", 1)
show_img_with_matplotlib(image_contours, "RETR EXTERNAL (None)", 2)
show_img_with_matplotlib(image_contours_2, "RETR LIST (Simple)", 3)
show_img_with_matplotlib(image_contours_3, "RETR LIST (TC89 KCos)", 4)
show_img_with_matplotlib(image_contours_4, "RETR LIST (TC89 L1)", 5)

plt.show()