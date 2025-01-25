import os

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template
from ultralytics import YOLO

app = Flask(__name__)

# Initialize YOLO models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
models_dir = os.path.join(os.path.dirname(__file__), 'models')
person_model_path = os.path.join(models_dir, 'yolov8s.pt')
ppe_model_path = os.path.join(models_dir, 'ovu.pt')

# Load models
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

    # Load ROI coordinates
    utils_dir = os.path.dirname(__file__)
    roi_coord_path = os.path.join(utils_dir, 'ROI_coord.txt')
    with open(roi_coord_path, "r") as f:
        coord = f.read().split()
    ROI_box = np.array([coord[0], coord[1], coord[2], coord[3]], dtype=int)

    # Process with person model
    person_results = person_model(frame, device=device)
    person_result = person_results[0]
    person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
    person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
    person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")
    person_indices = np.where(person_classes == 0)[0]
    person_bboxes = person_bboxes[person_indices]
    person_scores = person_scores[person_indices]

    # Process with PPE model
    ppe_results = ppe_model(frame, device=device, imgsz=640, conf=0.8, iou=0.4)
    ppe_result = ppe_results[0]
    ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
    ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
    ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")

    # Draw detections and process results
    for person_bbox, person_score in zip(person_bboxes, person_scores):
        wearing_helmet = False
        wearing_vest = False

        # Check for PPE
        for ppe_bbox, ppe_class in zip(ppe_bboxes, ppe_classes):
            overlap_ratio = calculate_overlap(ppe_bbox, person_bbox)
            if overlap_ratio > overlap_threshold:
                if ppe_class == 0:
                    wearing_helmet = True
                elif ppe_class == 1:
                    wearing_vest = True

        # Draw person detection
        color = (0, 255, 0) if (wearing_helmet and wearing_vest) else (0, 0, 255)
        (px1, py1, px2, py2) = person_bbox
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        label = "PPE" if (wearing_helmet and wearing_vest) else "No PPE"
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    return frame

def generate_frames():
    # Use your video file path
    video_path = os.path.join(os.path.dirname(__file__), 'BM02.mp4')
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError('Could not start camera.')

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        # Process frame
        processed_frame = process_frame(frame)
        
        # Convert to jpg
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
