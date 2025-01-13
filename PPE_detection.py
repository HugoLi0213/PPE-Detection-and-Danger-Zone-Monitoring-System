import os

import cv2
import numpy as np
from ultralytics import YOLO

# Paths to input image and pre-trained models
image_path = os.path.join(os.path.dirname(__file__), 'video.mp4')  # Ensure the path is correct

# Update paths to point to the models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')
person_model_path = os.path.join(models_dir, 'yolov8s.pt')
ppe_model_path = os.path.join(models_dir, 'ovu.pt')

# Check if the model files exist
if not os.path.exists(person_model_path):
    raise FileNotFoundError(f"The person model file '{person_model_path}' does not exist.")
if not os.path.exists(ppe_model_path):
    raise FileNotFoundError(f"The PPE model file '{ppe_model_path}' does not exist.")

# Construct the path to the ROI_coord.txt file in the utils directory
utils_dir = os.path.dirname(__file__)
roi_coord_path = os.path.join(utils_dir, 'ROI_coord.txt')

# Check if the file exists
if not os.path.exists(roi_coord_path):
    raise FileNotFoundError(f"The file '{roi_coord_path}' does not exist. Please check the path.")

# Open the file
with open(roi_coord_path, "r") as f:
    coord = f.read().split()

# Convert coordinates to integers
ROI_box = np.array([coord[0], coord[1], coord[2], coord[3]], dtype=int)

# Initialize YOLO models
cap = cv2.VideoCapture(image_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file '{image_path}'")

person_model = YOLO(person_model_path)  # Person detection model
ppe_model = YOLO(ppe_model_path)  # Custom helmet detection model

# Function to calculate IoU based on the helmet bounding box
def calculate_overlap(object_box, area_box):
    # Unpack box coordinates
    ox1, oy1, ox2, oy2 = object_box
    ax1, ay1, ax2, ay2 = area_box

    # Calculate the intersection coordinates
    ix1 = max(ox1, ax1)
    iy1 = max(oy1, ay1)
    ix2 = min(ox2, ax2)
    iy2 = min(oy2, ay2)

    # Calculate intersection area
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    intersection_area = inter_width * inter_height

    # Calculate the area of the helmet bounding box
    object_area = (ox2 - ox1) * (oy2 - oy1)

    # Avoid division by zero
    if object_area == 0:
        return 0

    # Compute the overlap ratio (intersection / object area)
    overlap_ratio = intersection_area / object_area
    return overlap_ratio

# Threshold for overlap ratio to consider a helmet as "worn"
overlap_threshold = 0.4  # Adjust based on your requirements

ROI_threshold = 0.5
ROI_count_last = 0
ROI_count_current = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from video. Exiting...")
        break

    # Perform person detection
    person_results = person_model(frame, device="cpu")  # change device to cuda if you need
    person_result = person_results[0]
    person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
    person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
    person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")
    # Filter to include only persons (class ID = 0)
    person_indices = np.where(person_classes == 0)[0]
    person_bboxes = person_bboxes[person_indices]
    person_scores = person_scores[person_indices]

    # Perform PPE detection
    ppe_results = ppe_model(frame, device="cpu", imgsz=640, conf=0.8, iou=0.4)  # change device to cuda if you need
    ppe_result = ppe_results[0]
    ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
    ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
    ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")
    # Filter to include only helmets (class ID = 0 in custom model)
    helmet_indices = np.where(ppe_classes == 0)[0]
    helmet_bboxes = ppe_bboxes[helmet_indices]
    helmet_scores = ppe_scores[helmet_indices]
    # Filter to include only vest (class ID = 1 in custom model)
    vest_indices = np.where(ppe_classes == 1)[0]
    vest_bboxes = ppe_bboxes[vest_indices]
    vest_scores = ppe_scores[vest_indices]

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

        # ROI check
        if not (wearing_helmet and wearing_vest):
            overlap_ratio = calculate_overlap(person_bbox, ROI_box)
            if overlap_ratio > ROI_threshold:
                ROI_count_current += 1

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
        (vx1, vy1, vx2, vy2) = vest_bbox
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 255, 0), 2)
        cv2.putText(frame, "Vest", (vx1, vy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    # Draw ROI
    ROIx1, ROIy1, ROIx2, ROIy2 = ROI_box
    # ROI color changes
    ROI_color = (255, 255, 255) if (ROI_count_current == 0) else (255, 0, 255)
    cv2.rectangle(frame, (ROIx1, ROIy1), (ROIx2, ROIy2), ROI_color, 2)
    cv2.putText(frame, "ROI", (ROIx1, ROIy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, ROI_color, 2)
    print("ROI count", ROI_count_current)
    # send MQTT message only if ROI count grow larger
    if ROI_count_current > ROI_count_last:
        print("Send MQTT Message")
    ROI_count_last = ROI_count_current
    ROI_count_current = 0

    # Display the image with detections
    display = cv2.resize(frame, (1200, 720))
    cv2.imshow("frame", display)
    if cv2.waitKey(100) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
