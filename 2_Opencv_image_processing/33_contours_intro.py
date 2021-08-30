import numpy as np
import cv2
from matplotlib import pyplot as plt

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
    ax = plt.subplot(1, 3, pos)
    plt.imshow(img_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Contours introduction", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

canvas = np.zeros((640, 640, 3), dtype='uint8')

contours = get_one_contour()
print("Contour shape: '{}'".format(contours[0].shape))
print("'detected' contours: '{}'".format(len(contours)))

# Create multiple copies
image_cnts_points = canvas.copy()
image_cnts_outline = canvas.copy()
image_cnts_points_outline = canvas.copy()

# Draw only contour points
draw_contour_points(image_cnts_points, contours, (255, 0, 255))

# Draw only contour outline
draw_contour_outline(image_cnts_outline, contours, (0, 255, 255), 3)

# Draw both
draw_contour_outline(image_cnts_points_outline, contours, (255, 0, 0), 3)
draw_contour_points(image_cnts_points_outline, contours, (0, 0, 255))

show_img_with_matplotlib(image_cnts_points, "Contour points", 1)
show_img_with_matplotlib(image_cnts_outline, "Contour outline", 2)
show_img_with_matplotlib(image_cnts_points_outline, "Contour outline and points", 3)

plt.show()