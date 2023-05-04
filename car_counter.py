"""
The flow :

1. read the video
2. make model
3. define classnames
4. make tracker
5. define limits for line
6. empty list for total_count
7. iterate through every frame
8. Mask the frame
9. read the graphics
10. empty array for storing detections
11. take out results using model
12. iterate through results
13. grab the boxes and iterate through it, and take out the coordinates.
14. confidence score(integer)
15. append the coordinates in the empty array
16. stack them vertically
17. make tracking
18. make line
19. iterate through tracking
20. trakcing gives x1,x2,y1,y2,id .
21. find width and height
22. make box
23 put text on the boxes
24. find the center of the detection box
25. append the total count
26. put text of count.
"""


import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import * # This is for tracking.
from tracker import Tracker

# Capturing the video.
cap = cv2.VideoCapture("Videos/cars.mp4")

# Selecting a model.
model = YOLO("yolo_weights/yolov8l.pt")

# defining Class Names
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Applying Mask, so that we can detect cars in a region only.
# We can make masks in canva.com

# Read the mask 
mask = cv2.imread("mask1.png")

# print(mask.shape)
# Tracker
#tracker = Sort(max_age=20 , min_hits=3,iou_threshold=0.3)
tracker = Tracker()

# Limits to make lines
limits = [400, 297, 673, 297]

# Count the total cars.
total_count = []

# Iterating through every single frame.
while True:
    # Reading each frame.
    success , img = cap.read()
    print(img.shape)

    # Applying mask on the frame.
    imgRegion = cv2.bitwise_and(img,mask)

    # Adding banner to show count on top of that.
    imgGraphics = cv2.imread("graphics.png",cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img,imgGraphics,(0,0))

    # Taking out results of the frame.
    results = model(imgRegion,stream=True)

    # Taking out detection from each frame.
    detections = np.empty((0,5))

    for r in results:
        detections2 = []
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1 , y2-y1

            # confidence score
            conf = math.ceil((box.conf[0]*100))/100

            # class_names
            classes = int(box.cls[0])
            currentClass = class_names[classes]

            if currentClass=="car" or currentClass=="truck" or currentClass=="bus" or currentClass=="motorbike" and conf > 0.5:
                # Storing detections in an array and appending them to the detection's empty array..
                # currentarray = np.array([x1,y1,x2,y2,conf])
                # detections = np.vstack((detections,currentarray))
                detections2.append([x1,y1,x2,y2,conf])

    # Updating the tracker.
    tracker.update(img,detections2)

    # Making the line.
    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),5)

    # Iterate through the results from the tracker.
    for results in tracker.tracks:
        x1,y1,x2,y2 = results.bbox
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        w,h = x2-x1 , y2-y1
        ID = int(results.track_id)
        cvzone.cornerRect(img, (x1, y1, w, h), l=1, rt=1,colorR=(255,0,255))
        cvzone.putTextRect(img, f'{currentClass} {ID}', (max(0, x1), max(35, y1)),scale=1, thickness=1, offset=3)

        # Making Circles in the center of the BBOX.
        cx, cy = x1+w // 2, y1+h // 2
        cv2.circle(img,(cx,cy),5,(255,0,255),cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-20 < cy < limits[3]+20:
            if total_count.count(ID)==0:
                total_count.append(ID)
                cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)

            # putting the count text of the graphic image.
    cv2.putText(img,str(len(total_count)),(255,100),cv2.FONT_HERSHEY_PLAIN,fontScale=3,color=(50,50,255),thickness=4)


    cv2.imshow("image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()