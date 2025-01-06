import cv2
from ultralytics import YOLO
import numpy as np

# Paths to input image and pre-trained models
image_path = "pic2.jpg"  # Change the image for detection
person_model_path = "yolov8s.pt"  
ppe_model_path = "ovu.pt"  # Replace with a model pre-trained for helmet detection


frame = cv2.imread(image_path)

# Initialize YOLO models
person_model = YOLO(person_model_path)  # Person detection model
ppe_model = YOLO(ppe_model_path)  # Custom helmet detection model

# Function to calculate IoU based on the helmet bounding box
def calculate_overlap(object_box, person_box):
    # Unpack box coordinates
    hx1, hy1, hx2, hy2 = object_box
    px1, py1, px2, py2 = person_box

    # Calculate the intersection coordinates
    ix1 = max(hx1, px1)
    iy1 = max(hy1, py1)
    ix2 = min(hx2, px2)
    iy2 = min(hy2, py2)

    # Calculate intersection area
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    intersection_area = inter_width * inter_height

    # Calculate the area of the helmet bounding box
    helmet_area = (hx2 - hx1) * (hy2 - hy1)

    # Avoid division by zero
    if helmet_area == 0:
        return 0

    # Compute the overlap ratio (intersection / helmet area)
    overlap_ratio = intersection_area / helmet_area
    return overlap_ratio

# Perform person detection
person_results = person_model(frame, device="cpu") #change device to cuda if you need
person_result = person_results[0]
person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")

# Filter to include only persons (class ID = 0)
person_indices = np.where(person_classes == 0)[0]
person_bboxes = person_bboxes[person_indices]
person_scores = person_scores[person_indices]

# Perform ppe detection
ppe_results = ppe_model(frame, device="cpu", imgsz=640, conf=0.8, iou=0.4) #change device to cuda if you need
ppe_result = ppe_results[0]
ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")

# Filter to include only helmets (class ID = 0 in custom model)
helmet_indices = np.where(ppe_classes == 0)[0]
helmet_bboxes = ppe_bboxes[helmet_indices]
helmet_scores = ppe_scores[helmet_indices]

# Filter to include only helmets (class ID = 1 in custom model)
vest_indices = np.where(ppe_classes == 1)[0]
vest_bboxes = ppe_bboxes[vest_indices]
vest_scores = ppe_scores[vest_indices]

# Threshold for overlap ratio to consider a helmet as "worn"
overlap_threshold = 0.4  # Adjust based on your requirements

# Check if persons are wearing helmets and vests
for person_bbox, person_score in zip(person_bboxes, person_scores):
    wearing_helmet = False  # Assume no helmet initially
    wearing_vest = False

    for helmet_bbox in helmet_bboxes:
        overlap_ratio = calculate_overlap(helmet_bbox, person_bbox)

        # If overlap ratio exceeds the threshold, the person is wearing a helmet
        if overlap_ratio > overlap_threshold:
            wearing_helmet = True
            break

    for vest_bbox in vest_bboxes:
        overlap_ratio = calculate_overlap(vest_bbox, person_bbox)
        if overlap_ratio > overlap_threshold:
            wearing_vest = True
            break

    # Draw bounding box around the person
    color = (0, 255, 0) if (wearing_helmet and wearing_vest) else (0, 0, 255)  # Green if wearing helmet and vest, red otherwise
    (px1, py1, px2, py2) = person_bbox
    cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
    label = "PPE" if wearing_helmet else "No PPE"
    cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

# Draw bounding boxes for detected helmets (for reference)
for helmet_bbox in helmet_bboxes:
    (hx1, hy1, hx2, hy2) = helmet_bbox
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)  # Blue for helmets
    cv2.putText(frame, "Helmet", (hx1, hy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

for vest_bbox in vest_bboxes:
    (hx1, hy1, hx2, hy2) = vest_bbox
    cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 255, 0), 2)
    cv2.putText(frame, "Helmet", (hx1, hy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

# Display the image with detections
framers = cv2.resize(frame,(1200,720))
cv2.imshow("Image", framers)
cv2.waitKey(0)
cv2.destroyAllWindows()