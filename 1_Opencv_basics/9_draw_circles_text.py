import cv2
import numpy as np

colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255), 'yellow': (0, 255, 255), 'magenta': (255, 0, 255), 'cyan': (255, 255, 0), 'dark_gray': (50, 50, 50)}

image = np.zeros((600, 600, 3), dtype='uint8')
# Change the background color
image[:] = colors['dark_gray']

def draw_text():

    # Set the position to be used for drawing text
    menu_pos = (10, 500)
    menu_pos2 = (10, 525)
    menu_pos3 = (10, 550)
    menu_pos4 = (10, 575)

    cv2.putText(image, 'Double left click: add a circle', menu_pos, cv2.FONT_HERSHEY_DUPLEX, 0.7, colors['red'], 1)
    cv2.putText(image, 'Single right click: delete last circle', menu_pos2, cv2.FONT_HERSHEY_DUPLEX, 0.7, colors['red'], 1)
    cv2.putText(image, 'Double right click: delete all circles', menu_pos3, cv2.FONT_HERSHEY_DUPLEX, 0.7, colors['red'], 1)
    cv2.putText(image, 'Press \'q\' to exit', menu_pos4, cv2.FONT_HERSHEY_DUPLEX, 0.7, colors['red'], 1)

def draw_circle(event, x, y, flags, param):

    global circles

    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("Add a circle")
        #cv2.circle(image, (x, y), 10, colors['magenta'], -1)
        circles.append((x, y))

    if event == cv2.EVENT_RBUTTONDBLCLK:
        print("Delete all circles")
        #cv2.circle(image, (x, y), 10, colors['magenta'], -1)
        circles[:] = []

    if event == cv2.EVENT_RBUTTONDOWN:
        print("Delete last added circle")
        try:
            circles.pop()
        except (IndexError):
            print("No circles to delete")

    if event == cv2.EVENT_MOUSEMOVE:
        print("event: EVENT_MOUSEMOVE")
    
    if event == cv2.EVENT_LBUTTONUP:
        print("event: EVENT_LBUTTONUP")

    if event == cv2.EVENT_LBUTTONDOWN:
        print("event: EVENT_LBUTTONDOWN")
    
circles = []

cv2.namedWindow("Image mouse")
cv2.setMouseCallback('Image mouse', draw_circle)
draw_text()

clone = image.copy()

while True:
    image = clone.copy()

    for pos in circles:
        cv2.circle(image, pos, 30, colors['blue'], -1)

    cv2.imshow('Image mouse', image)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()