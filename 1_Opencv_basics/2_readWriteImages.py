import cv2
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("input_image_path", help="This is the path to the image to be displayed")
parser.add_argument("output_image_path", help="This is the path to save the image at")
args = parser.parse_args()

# Load the image in two ways
image = cv2.imread(args.input_image_path)

#args = vars(parser.parse_args())
#image2 = cv2.imread(args["image_path"])

# Process the input image - convert it to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show the image on display
cv2.imshow("Loaded image", image)
cv2.imshow("Gray image", gray_image)

cv2.waitKey(0)
cv2.destroyAllWindows()