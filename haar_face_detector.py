import argparse
import imutils
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image",type=str,required=True,
help="path to the input image")
ap.add_argument("-c","--cascade",type=str,
                default="haarcascade_frontalface_default.xml",
                help="path to haar cascade face detector")
args=vars(ap.parse_args())

#load the haar cascade face detector from
print("[INFO] loading face detector....")
detector=cv2.CascadeClassifier(args["cascade"])


#load the input image from disk, resize it, and 
# convert it to grayscale
image=cv2.imread(args['image'])
image=imutils.resize(image,width=500)
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# detect faces in the input image using the haar
# cascade face detector

'''
    - scaleFactor: How much the image size is reduced 
        at each image scale. This value is used to create 
        the scale pyramid. To detect faces at multiple 
        scales in the image (some faces may be closer to 
        the foreground, and thus be larger, other faces 
        may be smaller and in the background, thus the 
        usage of varying scales). A value of 1.05 indicates
        that we are reducing the size of the image by 5% at 
        each level in the pyramid.

    - minNeighbors: How many neighbors each window should
        have for the area in the window to be considered a face. 
        The cascade classifier will detect multiple windows around 
        a face. This parameter controls how many rectangles (neighbors)
        need to be detected for the window to be labeled a face.

    - minSize: A tuple of width and height (in pixels) indicating 
        the window's minimum size. Bounding boxes smaller than 
        this size are ignored. It is a good idea to start with 
        (30, 30) and fine-tune from there.
'''
print("[INFO] Performing face detection...")
rects=detector.detectMultiScale(gray,scaleFactor=1.05,
                                minNeighbors=7,minSize=(30,30),
                                flags=cv2.CASCADE_SCALE_IMAGE)
print(f"[INFO] {len(rects)} faces detected...")

#loop over the bounding boxes
for (x,y,w,h) in rects:
    # draw the face bounding box on the image
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

file_name=args['image'].split('\\')[-1]

cv2.imshow("Image",image)
cv2.imwrite(f"Outputs/{file_name}",image)
cv2.waitKey(0)