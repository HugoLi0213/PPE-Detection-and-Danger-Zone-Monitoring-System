import multiprocessing as mp
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Paths to input image and pre-trained models
image_path = os.path.join(os.path.dirname(__file__), 'BM02.mp4')
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
device = 'cuda' if torch.cuda.is_available() else 'cpu'
person_model = YOLO(person_model_path).to(device)
ppe_model = YOLO(ppe_model_path).to(device)

def calculate_overlap(object_box, area_box):
    ox1, oy1, ox2, oy2 = object_box
    ax1, ay1, ax2, ay2 = area_box
    ix1 = max(ox1, ax1)
    iy1 = max(oy1, ay1)
    ix2 = min(ox2, ax2)
    iy2 = min(oy2, ay2)
    inter_width = max(0, ix2 - ix1)
    inter_height = max(0, iy2 - iy1)
    intersection_area = inter_width * inter_height
    object_area = (ox2 - ox1) * (oy2 - oy1)
    if object_area == 0:
        return 0
    overlap_ratio = intersection_area / object_area
    return overlap_ratio

def process_frame(frame):
    overlap_threshold = 0.4
    ROI_threshold = 0.5
    ROI_count_current = 0

    person_results = person_model(frame, device=device)
    person_result = person_results[0]
    person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
    person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
    person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")
    person_indices = np.where(person_classes == 0)[0]
    person_bboxes = person_bboxes[person_indices]
    person_scores = person_scores[person_indices]

    ppe_results = ppe_model(frame, device=device, imgsz=640, conf=0.8, iou=0.4)
    ppe_result = ppe_results[0]
    ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
    ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
    ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")
    helmet_indices = np.where(ppe_classes == 0)[0]
    helmet_bboxes = ppe_bboxes[helmet_indices]
    helmet_scores = ppe_scores[helmet_indices]
    vest_indices = np.where(ppe_classes == 1)[0]
    vest_bboxes = ppe_bboxes[vest_indices]
    vest_scores = ppe_scores[vest_indices]

    for person_bbox, person_score in zip(person_bboxes, person_scores):
        wearing_helmet = False
        wearing_vest = False

        for helmet_bbox in helmet_bboxes:
            overlap_ratio = calculate_overlap(helmet_bbox, person_bbox)
            if overlap_ratio > overlap_threshold:
                wearing_helmet = True
                break

        for vest_bbox in vest_bboxes:
            overlap_ratio = calculate_overlap(vest_bbox, person_bbox)
            if overlap_ratio > overlap_threshold:
                wearing_vest = True
                break

        if not (wearing_helmet and wearing_vest):
            overlap_ratio = calculate_overlap(person_bbox, ROI_box)
            if overlap_ratio > ROI_threshold:
                ROI_count_current += 1

        color = (0, 255, 0) if (wearing_helmet and wearing_vest) else (0, 0, 255)
        (px1, py1, px2, py2) = person_bbox
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        label = "PPE" if wearing_helmet else "No PPE"
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    for helmet_bbox in helmet_bboxes:
        (hx1, hy1, hx2, hy2) = helmet_bbox
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
        cv2.putText(frame, "Helmet", (hx1, hy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    for vest_bbox in vest_bboxes:
        (vx1, vy1, vx2, vy2) = vest_bbox
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 255, 0), 2)
        cv2.putText(frame, "Vest", (vx1, vy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    ROIx1, ROIy1, ROIx2, ROIy2 = ROI_box
    ROI_color = (255, 255, 255) if (ROI_count_current == 0) else (255, 0, 255)
    cv2.rectangle(frame, (ROIx1, ROIy1), (ROIx2, ROIy2), ROI_color, 2)
    cv2.putText(frame, "ROI", (ROIx1, ROIy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, ROI_color, 2)
    return frame, ROI_count_current

def main():
    cap = cv2.VideoCapture(image_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file '{image_path}'")

    ROI_count_last = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from video. Exiting...")
            break

        frame, ROI_count_current = process_frame(frame)

        if ROI_count_current > ROI_count_last:
            print("Send MQTT Message")
        ROI_count_last = ROI_count_current

        display = cv2.resize(frame, (1200, 720))
        cv2.imshow("frame", display)
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
