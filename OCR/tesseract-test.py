import cv2
import pytesseract as pts
import matplotlib.pyplot as plt
import numpy as np
from difflib import SequenceMatcher

def show_result(image, text):
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.text(0.0, 0.0, text, ha='left', va='bottom', size=10)

    plt.show()
#end show_result

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#end get_grayscale

def remove_noise(image):
    return cv2.medianBlur(image, 5)
#end remove_noise

def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#end thresholding

def dilate(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
#end dilate

def erode(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(image, kernel, iterations = 1)
#end erode

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
#end opening

def canny(image):
    return cv2.Canny(image, 100, 200)
#end canny

def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags = cv2.INTER_CUBIC, borderMode = cv2.BORDER_REPLICATE)
    return rotated
#end deskew

def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
#end match_template

img = cv2.imread('image.jpg')

gray = get_grayscale(img)
thresh = thresholding(gray)
opening = opening(gray)
canny = canny(gray)

"""
f, ax = plt.subplots(2, 2)
plt.axis('off')
ax[0, 0].imshow(gray)
ax[0, 0].set_title("gray")
ax[0, 1].imshow(thresh)
ax[0, 1].set_title("threshold")
ax[1, 0].imshow(opening)
ax[1, 0].set_title("opening")
ax[1, 1].imshow(canny)
ax[1, 1].set_title("canny")
plt.show()
"""

custom_config = r'--oem 3 --psm 6'
text_original = pts.image_to_string(img, config=custom_config)
text_grayscale = pts.image_to_string(gray, config=custom_config)
text_thresh = pts.image_to_string(thresh, config=custom_config)
text_opening = pts.image_to_string(opening, config=custom_config)
text_canny = pts.image_to_string(canny, config=custom_config)

"""
# show output text
print("\noriginal\n--------------------------\n" + text_original)
print("\ngrayscale\n--------------------------\n" + text_grayscale)
print("\nthreshold\n--------------------------\n" + text_thresh)
print("\nopening\n--------------------------\n" + text_opening)
print("\ncanny\n--------------------------\n" + text_canny)
"""

"""
# compare output text in different modes with original image
m = SequenceMatcher(None, text_original, text_grayscale)
print("original # grayscale\t %.03f\n" % (round(m.ratio(), 3)))

m = SequenceMatcher(None, text_original, text_thresh)
print("original # threshold\t %.03f\n" % (round(m.ratio(), 3)))

m = SequenceMatcher(None, text_original, text_opening)
print("original # openning\t %.03f\n" % (round(m.ratio(), 3)))

m = SequenceMatcher(None, text_original, text_canny)
print("original # canny\t %.03f\n" % (round(m.ratio(), 3)))
#show_result(img, text)
"""

h, w, c = img.shape
boxes = pts.image_to_boxes(img)
for b in boxes.splitlines():
    b = b.split(' ')
    img = cv2.rectangle(img, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

#cv2.imshow('img', img)
show_result(img, text_original)
cv2.waitKey(0)
