import cv2
import pytesseract as pts
import matplotlib.pyplot as plt
import numpy as np

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

custom_config = r'--oem 3 --psm 6'
text = pts.image_to_string(img, config=custom_config)

#print(text)
show_result(img, text)
