from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
        help="input image to be processed")
ap.add_argument("-r", "--reference", required=True,
        help="reference OCR-A font image")
args = vars(ap.parse_args())

# card type mapping with first digit of card number
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "Master",
    "6": "Discover Card"
}

# load reference image and preprocessing
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

#cv2.imshow("reference", ref)
#cv2.waitKey(0)


