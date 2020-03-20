from PIL import Image
import pytesseract as pts
import argparse
import cv2
import os

# parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="path to input OCR image")
ap.add_argument("-p", "--preprocess", type=str, default="thresh",
        help="type of preprocessing image to be done")
args = vars(ap.parse_args())

# load image and convert it to grayscale
image = cv2.imread(args["image"])
gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if args["preprocess"] == "thresh":
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
elif args["preprocess"] == "blue":
    gray = cv2.medianBlur(gray, 3)

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

text = pts.image_to_string(gray)
os.remove(filename)
print(text)

cv2.imshow("Image", image)
cv2.imshow("Output", gray)
cv2.waitKey(0)
