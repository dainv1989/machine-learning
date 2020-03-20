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

refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(type(refCnts)) # tuple
#print(refCnts)
refCnts = imutils.grab_contours(refCnts)
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

for (i, c) in enumerate(refCnts):
    # compute bounding box for the digit extract and resize it to fixed size
    (x, y, w, h) = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    # update digits dict, mapping to ROI
    digits[i] = roi
#end for (i, c)

#cv2.imshow("8", digits[8])
#cv2.waitKey(0)

