import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# load model
model = YOLO("ovu.pt")

# image path
path = "pic1.jpg"

# process
pic = cv2.imread(path)
results = model(pic,imgsz=640,conf=0.8,iou=0.4)
annotator = Annotator(pic)
for result in results:
    for box in result.boxes:
        annotator.box_label(box.xyxy[0], result.names[int(box.cls)])
pic = cv2.resize(pic,(1280,720))
cv2.imshow("frame", pic)
cv2.waitKey(0)
