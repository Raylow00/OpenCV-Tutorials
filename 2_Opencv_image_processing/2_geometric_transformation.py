import cv2
import numpy as np
import matplotlib.pyplot as plt

# This function shows an image using the matplotlib functionality
def show_with_matplotlib(color_img, title):
    # pos - position in the figure plot

    # First convert the BGR image to RGB
    rgb = color_img[:, :, ::-1] # all items in the array, reversed

    """
    Other slice notations worth noting:
    a[-2:]      # last two items in the array
    a[:-2]      # everything except the last two items
    a[1::-1]    # the first two items, reversed
    a[:-3:-1]   # the last two items, reversed
    a[-3::-1]   # everything except the last two items, reversed
    """

    #ax = plt.subplot(3, 6, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Read and show input image
image = cv2.imread('../../Images/tzuyu.jpg')
show_with_matplotlib(image, "Original image")

# Get the height and width of the image (remember OpenCV operations start with row by column)
height, width = image.shape[:2]

#-----------------------------------------------------
# 1. Scaling an image
# - by setting the specific end size
# - by providing the scale factors fx and fy
# To enlarge image, use cv2.INTER_CUBIC 
# To shrink image, use cv2.INTER_LINEAR
# The 5 interpolation methods provided by OpenCV:
# - cv2.INTER_NEAREST (Nearest neighbor interpolation)
# - cv2.INTER_LINEAR (Bilinear interpolation)
# - cv2.INTER_AREA (Resampling using pixel area relation)
# - cv2.INTER_CUBIC (Bicubic interpolation)
# - cv2.INTER_LANCZOS4 (Sinusoidal interpolation)

dst_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

dst_image_2 = cv2.resize(image, (width * 2, height * 2), interpolation=cv2.INTER_LINEAR)

show_with_matplotlib(dst_image, "Resized image with scale factor")
show_with_matplotlib(dst_image_2, "Resized image with specific size")

#-----------------------------------------------------
# 2. Translation (Moving the image to the right, left, top and bottom)
# We need to create the 2x3 or 3x3 transformation matrix using numpy array with float values (float32)
# Translation in the x direction: 200 px, Translation in the y direction: 30 px
# cv2.warpAffine - applies a transformation matrix to the input image
M = np.float32([[1, 0, 200], [0, 1, 30]])
dst_image = cv2.warpAffine(image, M, (width, height))

show_with_matplotlib(dst_image, "Translated image (positive values)")

M = np.float32([[1, 0, -200], [0, 1, -30]])
dst_image = cv2.warpAffine(image, M, (width, height))

show_with_matplotlib(dst_image, "Translated image (negative values)")

#-----------------------------------------------------
# 3. Rotation
# We create the 2x3 transformation matrix using cv2.getRotationMatrix2D()
# Params:
# - center
# - angle
# - scale
# Let's rotate the image 180 degrees upside down
M = cv2.getRotationMatrix2D((width/2.0, height/2.0), 180, 1)
dst_image = cv2.warpAffine(image, M, (width, height))

# Show the center of rotation and the rotated image
cv2.circle(dst_image, (round(width/2.0), round(height/2.0)), 5, (255, 0, 0), -1)
show_with_matplotlib(dst_image, "Image rotated 180 deg")

#-----------------------------------------------------
# 4. Affine Transformation
# We use the function cv2.getAffineTransform() to build the 2x3 transformation matrix
# which is obtained from the relation between three points 
# from the input image and the corresponding coordinates in the transformed image

image_copy = image.copy()
cv2.circle(image_copy, (135, 45), 5, (255, 0, 255), -1)
cv2.circle(image_copy, (385, 45), 5, (255, 0, 255), -1)
cv2.circle(image_copy, (135, 230), 5, (255, 0, 255), -1)

# Show the image with the three created points
show_with_matplotlib(image_copy, 'Before affine transformation')

# Then we create the arrays with the three points and the desired positions in the output image
pts_1 = np.float32([[135, 45], [385, 45], [135, 230]])
pts_2 = np.float32([[135, 45], [385, 45], [150, 230]])
M = cv2.getAffineTransform(pts_1, pts_2)
dst_image = cv2.warpAffine(image_copy, M, (width, height))

show_with_matplotlib(dst_image, "After affine transformation")

#-----------------------------------------------------
# 5. Perspective Transformation
# Requires 4 pairs of points to calculate a perspective transformation
image_copy = image.copy()
cv2.circle(image_copy, (450, 65), 5, (255, 0, 255), -1)
cv2.circle(image_copy, (517, 65), 5, (255, 0, 255), -1)
cv2.circle(image_copy, (431, 164), 5, (255, 0, 255), -1)
cv2.circle(image_copy, (552, 164), 5, (255, 0, 255), -1)

show_with_matplotlib(image_copy, "Before perspective transformation")

pts_1 = np.float32([[450, 65], [517, 65], [431, 164], [552, 164]])
pts_2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
M = cv2.getPerspectiveTransform(pts_1, pts_2)
print("Perspective transformation matrix shape: {}".format(M.shape))
dst_image = cv2.warpPerspective(image_copy, M, (300, 300))

show_with_matplotlib(dst_image, "After perspective transformation")

#-----------------------------------------------------
# 6. Cropping
# We use Numpy slicing to crop an image
image_copy = image.copy()

cv2.circle(image_copy, (230, 80), 5, (0, 0, 255), -1)
cv2.circle(image_copy, (330, 80), 5, (0, 0, 255), -1)
cv2.circle(image_copy, (230, 200), 5, (0, 0, 255), -1)
cv2.circle(image_copy, (330, 200), 5, (0, 0, 255), -1)
cv2.line(image_copy, (230, 80), (330, 80), (0, 0, 255))
cv2.line(image_copy, (330, 80), (330, 200), (0, 0, 255))
cv2.line(image_copy, (330, 200), (230, 200), (0, 0, 255))
cv2.line(image_copy, (230, 200), (230, 80), (0, 0, 255))

show_with_matplotlib(image_copy, "Before cropping")

dst_image = image[80:200, 230:330]

show_with_matplotlib(dst_image, "After cropping")


