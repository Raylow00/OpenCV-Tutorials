import cv2
import numpy as np
import matplotlib.pyplot as plt

# This function shows an image using the matplotlib functionality
def show_with_matplotlib(color_img, title, pos):
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

    ax = plt.subplot(3, 6, pos)
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')

# Load the original image
image = cv2.imread('../../Images/tzuyu.jpg')

# Create a figure() object with appropriate size and title
plt.figure(figsize=(16, 9))
plt.suptitle("Splitting and merging channels in OpenCV", fontsize=14, fontweight="bold")

# Show the BGR image
show_with_matplotlib(image, "BGR-image", 1)

#-----------------------------------------------------

# Split the image into the three channels
(b, g, r) = cv2.split(image)

# Show the three channels from the split image
show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR-B", 2)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR-G", 2+6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR-R", 2+6*2)

#-----------------------------------------------------

# Merge the three channel again to build the original image
image_copy = cv2.merge((b, g, r))

# Show the merged image
show_with_matplotlib(image_copy, "BGR-Merged", 1+6)

#-----------------------------------------------------

# Unless it is necessary, do not use cv2.split() as it is time consuming
# Use numpy indexing 
b_copy = image[:, :, 0]     # remember the last element returns the channel depth, here the first channel, B

# Let's see how to eliminate the blue channel
image_without_blue = image.copy()
# We eliminate (set to 0) the blue component
image_without_blue[:,:,0] = 0

# Eliminating the green channel
image_without_green = image.copy()
image_without_green[:,:,1] = 0

# Eliminating the red channel
image_without_red = image.copy()
image_without_red[:,:,2] = 0

# Show the three channels from the split image using numpy indexing
show_with_matplotlib(image_without_blue, "BGR without Blue", 3)
show_with_matplotlib(image_without_green, "BGR without Green", 3+6)
show_with_matplotlib(image_without_red, "BGR without Red", 3+6*2)

#-----------------------------------------------------

# Split the 'image_without_blue' into its three components
(b, g, r) = cv2.split(image_without_blue)

show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without Blue (B)", 4)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without Blue (G)", 4+6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without Blue (R)", 4+6*2)

#-----------------------------------------------------

# Split the 'image_without_green' into its three components
(b, g, r) = cv2.split(image_without_green)

show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without Green (B)", 5)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without Green (G)", 5+6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without Green (R)", 5+6*2)

#-----------------------------------------------------

# Split the 'image_without_red' into its three components
(b, g, r) = cv2.split(image_without_red)

show_with_matplotlib(cv2.cvtColor(b, cv2.COLOR_GRAY2BGR), "BGR without Red (B)", 6)
show_with_matplotlib(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), "BGR without Red (G)", 6+6)
show_with_matplotlib(cv2.cvtColor(r, cv2.COLOR_GRAY2BGR), "BGR without Red (R)", 6+6*2)

plt.show()
