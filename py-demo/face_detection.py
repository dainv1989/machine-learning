import sys
import cv2

if len(sys.argv) >= 2:
    imgPath = sys.argv[1]
else:
    print("please provide image path")
    quit()

# read image and process in gray color mode
image = cv2.imread(imgPath)
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# execute face detection
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

faces = faceCascade.detectMultiScale(
    imgray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
)

# draw retangles around found faces
print("found {0} faces".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (234, 67, 53), 2)

# show result image
cv2.imshow("faces found", image)
cv2.waitKey(0)
