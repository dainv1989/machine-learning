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

# init rectangle and square structuring kernels
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("gray", gray)
#cv2.waitKey(0)

# tophat (whitehat) morphological operator to find light regions against
# dark background
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
#cv2.imshow("tophat", tophat)
#cv2.waitKey(0)

# comput Scharr gradient then scale to range (0, 255)
gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
gradX = gradX.astype('uint8')

#cv2.imshow("gradX", gradX)
#cv2.waitKey(0)

# closing operation to close gap between credit card number digits
# then apply Otsu's thresholding method
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
#cv2.imshow("gradX", gradX)
#cv2.waitKey(0)

thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cv2.imshow("thresh", gradX)
#cv2.waitKey(0)

# apply the second closing operation
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
#cv2.imshow("thresh2", gradX)
#cv2.waitKey(0)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
locs = []

for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)

    # prune potential group of digit based on ratio
    if ar > 2.5 and ar < 4.0:
        if (w > 40 and w < 55) and (h > 10 and h < 20):
            locs.append((x, y, w, h))
            #tmp = cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

#cv2.imshow("tmp", tmp)
#cv2.waitKey(0)

locs = sorted(locs, key=lambda x:x[0])
output = []

for (i, (gX, gY, gW, gH)) in enumerate(locs):
    groupOutput = []

    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    digitCnts = imutils.grab_contours(digitCnts)
    digitCnts = contours.sort_contours(digitCnts, method="left-to-right")[0]

    for c in digitCnts:
        (x, y, w, h) = cv2.boundingRect(c)
        roi = group[y:y+h, x:x+w]
        roi = cv2.resize(roi, (57, 88))

        # init template of matching scores
        scores = []

        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(result)
            scores.append(score)

        groupOutput.append(str(np.argmax(scores)))

    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 255, 0), 2)
    cv2.putText(image, "".join(groupOutput), (gX, gY -15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    output.extend(groupOutput)

print("Credit card type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit card #: {}".format("".join(output)))
cv2.imshow("image", image)
cv2.waitKey(0)

