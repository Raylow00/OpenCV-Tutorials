import cv2
import color_constant
import numpy as np

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'dark_gray': (50, 50, 50)}

image = np.zeros((500, 500, 3), dtype='uint8')

# Change the background color
image[:] = colors['dark_gray']
cv2.line(image, (0, 0), (400, 400), colors['green'], 3)
cv2.line(image, (0, 400), (400, 0), colors['blue'], 3)
cv2.line(image, (200, 0), (200, 400), colors['red'], 3)
cv2.line(image, (0, 200), (400, 200), colors['yellow'], 3)

cv2.rectangle(image, (0, 0), (400, 400), colors['cyan'], 3)
cv2.rectangle(image, (150, 150), (350, 300), colors['red'], 1)

cv2.circle(image, (50, 50), 20, colors['green'], 3)
cv2.circle(image, (400, 200), 30, colors['magenta'], -1)

cv2.arrowedLine(image, (50, 50), (250, 50), colors['green'], 3, cv2.LINE_AA, 0, 0.3)
cv2.arrowedLine(image, (34, 34), (200, 200), colors['magenta'], 2, 8, 0, 1)

cv2.ellipse(image, (60, 60), (80, 80), 0, 0, 360, colors['yellow'], 3)
cv2.ellipse(image, (200, 200), (20, 40), 45, 0, 360, colors['blue'], 3)

# Drawing polygons using polylines with the parameter pts in the shape of (number_vertex, 1, 2)
# isClosed
# color
# thickness
pts = np.array([[250, 5], [220, 80], [280, 90]], np.int32)
# Reshape the shape to (number_vertex, 1, 2)
pts = pts.reshape((-1, 1, 2))
cv2.polylines(image, [pts], True, colors['green'], 3)

# Shift parameter in drawing
shift = 2
factor = 2 ** shift
print("Factor: {}".format(factor))
cv2.circle(image, (int(round(299.99 * factor)), int(round(299.99 * factor))), 300 * factor, colors['red'], 1)
cv2.circle(image, (299, 299), 300, colors['green'], 1)

# Draw text
cv2.putText(image, 'Mastering OpenCV4 with Python', (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, colors['green'], 2)
cv2.putText(image, 'Mastering OpenCV4 with Python', (10, 360), cv2.FONT_HERSHEY_COMPLEX, 1, colors['green'], 2)
cv2.putText(image, 'Mastering OpenCV4 with Python', (10, 390), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, colors['green'], 2)
cv2.putText(image, 'Mastering OpenCV4 with Python', (10, 420), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, colors['green'], 2)



cv2.imshow("Image", image)

while True:
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()