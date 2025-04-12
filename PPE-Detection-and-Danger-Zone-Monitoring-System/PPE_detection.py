# Move these to the very top of the file, before all other imports
import eventlet

eventlet.monkey_patch()

# Then continue with other imports
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

# Flask app initialization with SocketIO
app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Paths to input image and pre-trained models
image_path = os.path.join(os.path.dirname(__file__), 'test2.mp4')
models_dir = os.path.join(os.path.dirname(__file__), 'models')
person_model_path = os.path.join(models_dir, 'yolov8s.pt')
ppe_model_path = os.path.join(models_dir, 'ovu.pt')

# Check if the model files exist
if not os.path.exists(person_model_path):
    raise FileNotFoundError(f"The person model file '{person_model_path}' does not exist.")
if not os.path.exists(ppe_model_path):
    raise FileNotFoundError(f"The PPE model file '{ppe_model_path}' does not exist.")

# Construct the path to the ROI_coord.txt file
utils_dir = os.path.dirname(__file__)
roi_coord_path = os.path.join(utils_dir, 'ROI_coord.txt')

# Create ROI_coord.txt if it doesn't exist
if not os.path.exists(roi_coord_path):
    with open(roi_coord_path, "w") as f:
        f.write("0 0 640 480")  # Default coordinates

# Load initial ROI coordinates
with open(roi_coord_path, "r") as f:
    coord = f.read().split()
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

# Global variables to store PPE statistics
notification_active = False
notification_timestamp = None
notification_count = 0
ppe_stats = {"with_ppe": 0, "without_ppe": 0, "total_persons": 0, "date": ""}

def save_screenshot(frame):
    screenshot_dir = os.path.join(os.path.dirname(__file__), 'static', 'screenshots')
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_path = os.path.join(screenshot_dir, f"screenshot_{timestamp}.jpg")
    cv2.imwrite(screenshot_path, frame)
    print(f"Screenshot saved: {screenshot_path}")

last_alert_time = None
def process_frame(frame):
    global ppe_stats, ROI_box, notification_active, notification_timestamp, notification_count,last_alert_time
    overlap_threshold = 0.4
    ROI_threshold = 0.5
    roi_has_unsafe = False  # Track overall ROI safety status


    # Reload ROI coordinates
    with open(roi_coord_path, "r") as f:
        coord = f.read().split()
        if len(coord) >= 4:
            ROI_box = np.array([int(coord[0]), int(coord[1]), 
                               int(coord[2]), int(coord[3])], dtype=int)

    # Person detection
    person_results = person_model(frame, device=device)
    person_result = person_results[0]
    person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
    person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
    person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")
    person_indices = np.where(person_classes == 0)[0]
    person_bboxes = person_bboxes[person_indices]
    person_scores = person_scores[person_indices]

    ppe_stats["total_persons"] = len(person_bboxes)

    # PPE detection
    ppe_results = ppe_model(frame, device=device, imgsz=640, conf=0.8, iou=0.4)
    ppe_result = ppe_results[0]
    ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
    ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
    ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")
    
    helmet_bboxes = ppe_bboxes[np.where(ppe_classes == 0)[0]]
    vest_bboxes = ppe_bboxes[np.where(ppe_classes == 1)[0]]

    # Reset PPE stats
    ppe_stats["with_ppe"] = 0
    ppe_stats["without_ppe"] = 0

    for person_bbox, person_score in zip(person_bboxes, person_scores):
        wearing_helmet = False
        wearing_vest = False

        # Check helmet overlap
        for helmet_bbox in helmet_bboxes:
            if calculate_overlap(helmet_bbox, person_bbox) > overlap_threshold:
                wearing_helmet = True
                break

        # Check vest overlap
        for vest_bbox in vest_bboxes:
            if calculate_overlap(vest_bbox, person_bbox) > overlap_threshold:
                wearing_vest = True
                break

        # Update PPE stats
        if wearing_helmet and wearing_vest:
            ppe_stats["with_ppe"] += 1
        else:
            ppe_stats["without_ppe"] += 1

        # Check ROI overlap and safety
        if calculate_overlap(person_bbox, ROI_box) > ROI_threshold:
            if not (wearing_helmet and wearing_vest):
                roi_has_unsafe = True
                current_time = datetime.now()

        # Draw person bounding box
        color = (0, 255, 0) if (wearing_helmet and wearing_vest) else (0, 0, 255)
        px1, py1, px2, py2 = person_bbox
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        label = "PPE" if wearing_helmet and wearing_vest else "No PPE"
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # Draw PPE equipment boxes
    for h_bbox in helmet_bboxes:
        x1, y1, x2, y2 = h_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Helmet", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    for v_bbox in vest_bboxes:
        x1, y1, x2, y2 = v_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(frame, "Vest", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    # Update ROI display and capture
    ROI_color = (0, 255, 0) if not roi_has_unsafe else (0, 0, 255)
    ROI_label = "Safety" if not roi_has_unsafe else "Not Safety"
    x1, y1, x2, y2 = ROI_box
    cv2.rectangle(frame, (x1, y1), (x2, y2), ROI_color, 2)
    cv2.putText(frame, ROI_label, (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, ROI_color, 2)

    # Trigger notification if unsafe condition detected
    if roi_has_unsafe:
        if last_alert_time is None or (current_time - last_alert_time).total_seconds() > 3:  
            last_alert_time = current_time
            save_screenshot(frame)
            notification_active = True
            notification_timestamp = datetime.now()
            notification_count += 1
            
            # Emit WebSocket event with alert data
            alert_data = {
                'active': True,
                'timestamp': notification_timestamp.isoformat(),
                'count': notification_count,
                'message': 'Person without proper PPE detected in danger zone'
            }
            socketio.emit('safety_alert', alert_data)

    # Update timestamp
    ppe_stats["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return frame, roi_has_unsafe

last_danger_alert_time = None
def process_danger_zone_frame(frame):
    global ROI_box, notification_active, notification_timestamp, notification_count,last_danger_alert_time
    overlap_threshold = 0.4
    ROI_threshold = 0.5
    ROI_count_current = 0
    danger_zone_has_person = False  # Track if anyone is in the danger zone

    # Reload ROI coordinates
    with open(roi_coord_path, "r") as f:
        coord = f.read().split()
        if len(coord) >= 4:
            ROI_box = np.array([int(coord[0]), int(coord[1]), 
                            int(coord[2]), int(coord[3])], dtype=int)

    # Only detect people for danger zone monitoring
    person_results = person_model(frame, device=device)
    person_result = person_results[0]
    person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
    person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
    person_indices = np.where(person_classes == 0)[0]
    person_bboxes = person_bboxes[person_indices]

    for person_bbox in person_bboxes:
        px1, py1, px2, py2 = person_bbox
        
        # Calculate overlap with ROI
        overlap_ratio = calculate_overlap(person_bbox, ROI_box)
        if overlap_ratio > ROI_threshold:
            ROI_count_current += 1
            danger_zone_has_person = True
            current_time = datetime.now()
            color = (0, 0, 255)  # Red for person in danger zone
        else:
            color = (0, 255, 0)  # Green for safe

        # Draw person detection
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        status = "IN DANGER ZONE" if overlap_ratio > ROI_threshold else "Safe"
        cv2.putText(frame, status, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
            
    # Draw ROI
    ROIx1, ROIy1, ROIx2, ROIy2 = ROI_box
    ROI_color = (255, 0, 0) if ROI_count_current > 0 else (255, 255, 255)
    cv2.rectangle(frame, (ROIx1, ROIy1), (ROIx2, ROIy2), ROI_color, 2)
    cv2.putText(frame, f"Danger Zone (People: {ROI_count_current})", 
                (ROIx1, ROIy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, ROI_color, 2)

    # Trigger notification if people are detected in danger zone
    if danger_zone_has_person:
        if last_danger_alert_time is None or (current_time - last_danger_alert_time).total_seconds() > 3:
            last_danger_alert_time = current_time
            save_screenshot(frame)
            notification_active = True
            notification_timestamp = datetime.now()
            notification_count += 1
            
            # Emit WebSocket event with alert data
            alert_data = {
                'active': True,
                'timestamp': notification_timestamp.isoformat(),
                'count': notification_count,
                'message': 'Person detected in danger zone'
            }
            socketio.emit('safety_alert', alert_data)
    return frame, ROI_count_current

def generate_frames():
    cap = cv2.VideoCapture(image_path)
    if not cap.isOpened():
        raise RuntimeError('Could not start camera.')

    ROI_count_last = 0
    frame_count = 0  # Add a frame counter

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        processed_frame, ROI_count_current = process_frame(frame)
        
        # Add a yield point every few frames
        frame_count += 1
        if frame_count % 5 == 0:
            eventlet.sleep(0.01)  # Give the event loop a chance to run
        
        if ROI_count_current > ROI_count_last:
            print("Send MQTT Message")
        ROI_count_last = ROI_count_current

        # Resize frame for display
        display = cv2.resize(processed_frame, (1200, 720))
        
        # Convert to jpg for streaming
        ret, buffer = cv2.imencode('.jpg', display)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_danger_zone_frames():
    cap = cv2.VideoCapture(image_path)
    if not cap.isOpened():
        raise RuntimeError('Could not start camera.')

    ROI_count_last = 0
    frame_count = 0  # Add a frame counter

    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue

        processed_frame, ROI_count_current = process_danger_zone_frame(frame)
        
        # Add a yield point every few frames
        frame_count += 1
        if frame_count % 5 == 0:
            eventlet.sleep(0.01)  # Give the event loop a chance to run
        
        if ROI_count_current > ROI_count_last:
            print("Send Danger Zone Alert MQTT Message")
        ROI_count_last = ROI_count_current

        # Resize frame for display
        display = cv2.resize(processed_frame, (1200, 720))
        
        # Convert to jpg for streaming
        ret, buffer = cv2.imencode('.jpg', display)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/danger_zone_feed')
def danger_zone_feed():
    return Response(generate_danger_zone_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/manage')
def manage():
    cap = cv2.VideoCapture(image_path)
    success, frame = cap.read()
    if success:
        img_oh = frame.shape[0]  # original height
        img_ow = frame.shape[1]  # original width
        cap.release()
        return render_template('manage.html', img_ow=img_ow, img_oh=img_oh)
    return "Error loading video", 500

@app.route('/get_initial_frame')
def get_initial_frame():
    cap = cv2.VideoCapture(image_path)
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (1200, 720))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        cap.release()
        return Response(frame_bytes, mimetype='image/jpeg')
    return "Error loading frame", 500

@app.route('/get_coordinates')
def get_coordinates():
    try:
        with open(roi_coord_path, "r") as f:
            coordinates = f.read().split()
            return jsonify({"coordinates": coordinates})
    except:
        return jsonify({"coordinates": []})

@app.route('/add_coordinate', methods=['POST'])
def add_coordinate():
    data = request.json
    x, y = data['x'], data['y']
    with open(roi_coord_path, "a") as f:
        f.write(f"{x} {y} ")
    return jsonify({"status": "success"})

@app.route('/clear_coordinates', methods=['POST'])
def clear_coordinates():
    with open(roi_coord_path, "w") as f:
        f.write("")
    return jsonify({"status": "success"})

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/get_ppe_stats')
def get_ppe_stats():
    return jsonify(ppe_stats)

@app.route('/screenshots')
def show_screenshots():
    screenshot_dir = os.path.join(os.path.dirname(__file__), 'static', 'screenshots')
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('.jpg')]
    return render_template('screenshots.html', screenshots=screenshots)

@app.route('/api/notifications/clear', methods=['POST'])
def clear_notifications():
    global notification_active, notification_count
    notification_active = False
    notification_count = 0
    
    # Emit WebSocket event to clear notifications
    socketio.emit('clear_alerts', {'success': True})
    
    return jsonify({'success': True})

@app.route('/api/notifications/status')
def notification_status():
    global notification_active, notification_timestamp, notification_count
    return jsonify({
        'active': notification_active,
        'timestamp': notification_timestamp.isoformat() if notification_timestamp else None,
        'count': notification_count
    })

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send current notification status on connect
    emit('safety_alert', {
        'active': notification_active,
        'timestamp': notification_timestamp.isoformat() if notification_timestamp else None,
        'count': notification_count
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, debug=True)
