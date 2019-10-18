# references:
#   https://realpython.com/face-recognition-with-python/
#
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

# NOTE:
# each set of parameter values work best for individual image
faces = faceCascade.detectMultiScale(
    imgray,
    scaleFactor=1.05,
    minNeighbors=5,
    minSize=(20,20),                    # increase this size cause less detected faces
    flags=cv2.CASCADE_SCALE_IMAGE
)

# draw retangles around found faces
# color codes are BGR instead of RGB
print("found {0} faces".format(len(faces)))
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (234, 67, 53), 2)

# show result image
cv2.imshow("faces found", image)
cv2.waitKey(0)

imgOutput = "output.jpg"
cv2.imwrite(imgOutput, image)

cv2.destroyAllWindows()
