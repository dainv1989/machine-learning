# references:
#   https://realpython.com/face-detection-in-python-using-a-webcam/
import sys
import cv2

videoFile = ""

if len(sys.argv) == 2:
    videoFile = sys.argv[1]
else:
    print("No video input. Capturing from camera")

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

if videoFile != "":
    inVideo = cv2.VideoCapture(videoFile)
else:
    # 0 is camera index
    # in case multiple cameras connected, this index to choose camera
    inVideo = cv2.VideoCapture(0)

# define video writer
outFile = "outputVideo.avi"
resolution = (640, 360)
fps =  25.0
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#outVideo = cv2.VideoWriter(outFile, fourcc, fps, resolution)

while True:
    # capture frame by frame
    ret, frame = inVideo.read()

    if ret == True:
        # detect faces
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray_frame,
            scaleFactor = 1.05,
            minNeighbors = 2,
            minSize = (20, 20),
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (234, 67, 53), 2)

        cv2.imshow("Face detecting", frame)

        # save output video
        #frame = cv2.flip(frame, 0)
        #outVideo.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# release resource
inVideo.release()
#outVideo.release()
cv2.destroyAllWindows()
