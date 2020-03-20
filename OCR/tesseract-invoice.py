import cv2
import re
import pytesseract as pts
from pytesseract import Output

def show_textboxes(image):
    d = pts.image_to_data(image, output_type=Output.DICT)
    #print(d.keys())

    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            img = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image
#end show_textboxes

def find_pattern(image, data_pattern):
    d = pts.image_to_data(image, output_type=Output.DICT)
    keys = list(d.keys())
    n_boxes = len(d['text'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:
            if re.match(data_pattern, d['text'][i]):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image
#end find_pattern

img = cv2.imread('invoice.jpg')
#text = pts.image_to_string(img)
#print(text)

date_pattern = '^(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d\d$'
img_datebox = find_pattern(img, date_pattern)
#cv2.imshow('img', img_datebox)

email_pattern = '^[a-z0-9]+@[a-z0-9]+\.[a-z]+$'
img_emailbox = find_pattern(img_datebox, email_pattern)
cv2.imshow('img', img_emailbox)

#img_textboxes = show_textboxes(img)
#cv2.imshow('img', img_textboxes)
cv2.waitKey(0)
