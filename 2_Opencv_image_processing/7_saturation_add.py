import numpy as np
import cv2

# Take note that we are adding 8-bit values, results in values 0-255
# OpenCV addition is a saturated operation - returns a number limited to the minimum and maximum
# Numpy addition is a modulo operation - returns the remainder of a division
x = np.uint8([250])
y = np.uint8([50])

# In OpenCV addition, values are clipped to stay within range
# 250+50 = 300 => 255
result_opencv = cv2.add(x, y)
print("cv2.add(x: '{}', y:'{}') = '{}'".format(x, y, result_opencv))

# In Numpy addition, values are wrapped around
# 250+50 = 300 % 256 => 44
result_numpy = x + y
print("x: '{}', y: '{}' = '{}'".format(x, y, result_numpy))