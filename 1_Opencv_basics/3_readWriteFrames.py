import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("index_camera", help="Index of the camera to read from", type=int)
args = parser.parse_args()

# Video capture object
capture = cv2.VideoCapture(args.index_camera)

# Get properties of VideoCapture
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

print("CV-CAP-PROP-FRAME-WIDTH: '{}'".format(frame_width))
print("CV-CAP-PROP-FRAME-HEIGHT: '{}'".format(frame_height))
print("CV-CAP-PROP-FPS: '{}'".format(fps))

# Check if camera opened successfully
if capture.isOpened() is False:
    print("Error opening the camera")

# Read until video is completed
while capture.isOpened():
    ret, frame = capture.read()

    if ret is True:
        cv2.imshow("Feed", frame)

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the grayscale frame
        cv2.imshow("Gray feed", gray_frame)

        if cv2.waitKey(20) & 0xFF == 'q':
            break

    else:
        break

capture.release()
cv2.destroyAllWindows()

