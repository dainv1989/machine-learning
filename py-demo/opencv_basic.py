import cv2 as cv

capture = cv.VideoCapture(0)

while(True):
    ret, frame = capture.read()
    cv.imshow('video title', frame)

    # press ESC key to quit
    if cv.waitKey(1) == 27:
        break

capture.release()
cv.destroyAllWindows()
