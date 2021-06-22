import cv2
import argparse
import time

parser = argparse.ArgumentParser()

parser.add_argument("output_video_path", help="path to the video file to write")
args = parser.parse_args()

capture = cv2.VideoCapture(0)

# Video properties
frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = capture.get(cv2.CAP_PROP_FPS)

# Video codec to use
# FourCC is a 4-byte code used to specify the video codec
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
# or fourcc cv2.VideoWriter_fourcc(*'XVID')

out_gray = cv2.VideoWriter(args.output_video_path, fourcc, int(fps), (int(frame_width), int(frame_height)))

while capture.isOpened():
    ret, frame = capture.read()

    if ret is True:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        out_gray.write(gray_frame)

        cv2.imshow('gray', frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    else:
        break

capture.release()
out_gray.release()
cv2.destroyAllWindows()
