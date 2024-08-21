from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-c","--cascade",type=str,
                default="haarcascade_frontalface_default.xml",
                help="path to haar cascade face detector")

args=vars(ap.parse_args())

#load the haarcascade face detector from
print("[INFO] loading the face detector...")
detector=cv2.CascadeClassifier(args['cascade'])

# initialize the video stream and allow the camera sensor to
# warmup
print("[INFO] Starting video stream...")
vs=VideoStream(src=0).start()
time.sleep(2.0)

#loop over the frames from the video stream
while True:
    # grab the frame from the video stream, resize it
    # and convert to grayscale
    frame=vs.read()
    frame=imutils.resize(frame,width=500)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #perform face detection
    rects=detector.detectMultiScale(gray,
                                    scaleFactor=1.05,
                                    minNeighbors=5,
                                    minSize=(30,30),
                                    flags=cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in rects:
        # draw the face bounding box on the image
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1)&0xFF
    if key==ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()