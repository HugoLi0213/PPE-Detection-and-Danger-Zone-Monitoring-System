# Ensure eventlet monkey patching happens first
import eventlet

eventlet.monkey_patch()

# Now import other modules
import argparse
import os
from datetime import datetime

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request
from flask_socketio import SocketIO, emit
from pygrabber.dshow_graph import FilterGraph
from shapely.geometry import Polygon
from ultralytics import YOLO

# Flask app initialization
app = Flask(__name__, static_folder='static')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Input", default=None, help="Path to input image, video, or None for camera")
parser.add_argument("--model-version", choices=["v8", "v11"], default="v8", help="YOLO model version (v8 or v11)")
args = parser.parse_args()

# Settings
draw_helmet = 0
draw_vest = 0
text_name_format = "{:s}_{:s}.{:s}"
screenshot_name_format = "{:s}-{:%Y%m%d-%H%M%S}.{:s}"
camera_index = 0
input_type = ""
input_name = ""
img_rh = 720
img_rw = 1200

# Model settings
class ModelSettings:
    conf_threshold: float = 0.8
    iou_threshold: float = 0.4

# Determine input type
if args.Input is not None:
    if os.path.exists(args.Input):
        if args.Input.endswith((".png", ".jpg", ".jpeg")):
            input_type = "image"
            input_name = os.path.basename(args.Input)
        elif args.Input.endswith(".mp4"):
            input_type = "video"
            input_name = os.path.basename(args.Input)
        else:
            print("Wrong input format, need to be in [*.png, *.jpg, *.jpeg, *.mp4]")
            exit()
    else:
        print("Input not exist, exiting...")
        exit()
else:
    devices = FilterGraph().get_input_devices()
    if len(devices) != 0:
        input_type = "camera"
        input_name = devices[camera_index]
    else:
        print("No camera detected, exiting...")
        exit()

# Directories
models_dir = os.path.join(os.path.dirname(__file__), 'models')
detected_dir = os.path.join(os.path.dirname(__file__), 'detected')
zone_dir = os.path.join(os.path.dirname(__file__), 'zone')
for d in [detected_dir, zone_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# Model paths
person_model_path = os.path.join(models_dir, 'yolov11s.pt')
ppe_model_path = os.path.join(models_dir, 'ppe_v11s.pt')

# Check model files
if not os.path.exists(person_model_path):
    raise FileNotFoundError(f"The person model file '{person_model_path}' does not exist.")
if not os.path.exists(ppe_model_path):
    raise FileNotFoundError(f"The PPE model file '{ppe_model_path}' does not exist.")

# Zone file
zone_text_path = os.path.join(zone_dir, text_name_format.format("zone", input_name, "txt"))
if not os.path.exists(zone_text_path):
    with open(zone_text_path, "w"):
        pass

# Global zone variables
zone_list = []
zone_current_count = []
zone_last_count = []
haveZone = False

def load_zones():
    global zone_list, zone_current_count, zone_last_count, haveZone
    zone_list = []
    if os.path.exists(zone_text_path):
        with open(zone_text_path, "r") as f:
            zone_list = [eval(line.strip()) for line in f.readlines() if line.strip()]
    haveZone = len(zone_list) > 0
    zone_current_count = [0] * len(zone_list)
    zone_last_count = [0] * len(zone_list)

# Load zones initially
load_zones()

# Initialize YOLO models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
person_model = YOLO(person_model_path).to(device)
ppe_model = YOLO(ppe_model_path).to(device)

# Global variables
notification_active = False
notification_timestamp = None
notification_count = 0
ppe_stats = {"with_ppe": 0, "without_ppe": 0, "total_persons": 0, "date": ""}
last_alert_time = None

# Overlap functions
def PPE_overlap(object_box, area_box):
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
    return intersection_area / object_area if object_area > 0 else 0

def ZONE_overlap(person_box, zone_coord):
    px1, py1, px2, py2 = person_box
    person = Polygon([(px1, py1), (px2, py1), (px2, py2), (px1, py2)])
    zone = Polygon(zone_coord)
    if person.intersects(zone):
        return person.intersection(zone).area / person.area
    return 0

def save_screenshot(frame, zone_idx=None):
    screenshot_dir = os.path.join(os.path.dirname(__file__), 'static', 'screenshots')
    if not os.path.exists(screenshot_dir):
        os.makedirs(screenshot_dir)
    prefix = f"zone{zone_idx}" if zone_idx is not None else "alert"
    screenshot_name = screenshot_name_format.format(prefix, datetime.now(), "jpg")
    screenshot_path = os.path.join(screenshot_dir, screenshot_name)
    cv2.imwrite(screenshot_path, frame)
    return screenshot_name

def process_frame(frame, mode="ppe"):
    global ppe_stats, notification_active, notification_timestamp, notification_count, last_alert_time
    PPE_overlap_threshold = 0.4
    ZONE_overlap_threshold = 0.5
    zone_frame_arr = [frame.copy() for _ in zone_list]
    roi_has_unsafe = False

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
    ppe_stats["with_ppe"] = 0
    ppe_stats["without_ppe"] = 0

    # PPE detection
    ppe_results = ppe_model(
        frame, 
        device=device, 
        imgsz=640, 
        conf=ModelSettings.conf_threshold, 
        iou=ModelSettings.iou_threshold
    )
    ppe_result = ppe_results[0]
    ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
    ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
    ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")
    helmet_bboxes = ppe_bboxes[np.where(ppe_classes == 0)[0]]
    vest_bboxes = ppe_bboxes[np.where(ppe_classes == 1)[0]]

    for person_bbox, person_score in zip(person_bboxes, person_scores):
        wearing_helmet = False
        wearing_vest = False

        for helmet_bbox in helmet_bboxes:
            if PPE_overlap(helmet_bbox, person_bbox) > PPE_overlap_threshold:
                wearing_helmet = True
                break
        for vest_bbox in vest_bboxes:
            if PPE_overlap(vest_bbox, person_bbox) > PPE_overlap_threshold:
                wearing_vest = True
                break

        if wearing_helmet and wearing_vest:
            ppe_stats["with_ppe"] += 1
        else:
            ppe_stats["without_ppe"] += 1
            
        person_in_danger = False
        for i, zone_coord in enumerate(zone_list):
            overlap_ratio = ZONE_overlap(person_bbox, zone_coord)
            if overlap_ratio > ZONE_overlap_threshold:
                zone_current_count[i] += 1
                if not (wearing_helmet and wearing_vest) and mode == "ppe":
                    person_in_danger = True
                    roi_has_unsafe = True
                    cv2.rectangle(zone_frame_arr[i], (person_bbox[0], person_bbox[1]),
                                  (person_bbox[2], person_bbox[3]), (0, 0, 255), 2)
                    cv2.putText(zone_frame_arr[i], "No PPE", (person_bbox[0], person_bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                elif mode == "danger":
                    person_in_danger = True
                    cv2.rectangle(zone_frame_arr[i], (person_bbox[0], person_bbox[1]),
                                  (person_bbox[2], person_bbox[3]), (0, 0, 255), 2)
                    cv2.putText(zone_frame_arr[i], "IN DANGER ZONE", (person_bbox[0], person_bbox[1] - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        color = (0, 255, 0) if wearing_helmet and wearing_vest else (0, 0, 255)
        cv2.rectangle(frame, (person_bbox[0], person_bbox[1]), (person_bbox[2], person_bbox[3]), color, 2)
        label = "PPE" if wearing_helmet and wearing_vest else "No PPE"
        cv2.putText(frame, label, (person_bbox[0], person_bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    if draw_helmet:
        for h_bbox in helmet_bboxes:
            cv2.rectangle(frame, (h_bbox[0], h_bbox[1]), (h_bbox[2], h_bbox[3]), (255, 0, 0), 2)
            cv2.putText(frame, "Helmet", (h_bbox[0], h_bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    if draw_vest:
        for v_bbox in vest_bboxes:
            cv2.rectangle(frame, (v_bbox[0], v_bbox[1]), (v_bbox[2], v_bbox[3]), (255, 255, 0), 2)
            cv2.putText(frame, "Vest", (v_bbox[0], v_bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    for i, zone_coord in enumerate(zone_list):
        zone_color = (255, 0, 255) if zone_current_count[i] > 0 else (255, 255, 255)
        cv2.polylines(frame, [np.array(zone_coord, dtype=np.int32)], True, zone_color, 2)
        cv2.polylines(zone_frame_arr[i], [np.array(zone_coord, dtype=np.int32)], True, zone_color, 2)
        cv2.putText(frame, f"Zone {i} ({zone_current_count[i]})", (zone_coord[0][0], zone_coord[0][1] - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, zone_color, 2)

    if roi_has_unsafe or any(c > 0 for c in zone_current_count):
        current_time = datetime.now()
        if last_alert_time is None or (current_time - last_alert_time).total_seconds() > 3:
            last_alert_time = current_time
            notification_active = True
            notification_timestamp = current_time
            notification_count += 1
            message = "Person without PPE in zone" if mode == "ppe" else "Person in danger zone"
            for i, count in enumerate(zone_current_count):
                if count > zone_last_count[i]:
                    screenshot_name = save_screenshot(zone_frame_arr[i], i)
                    socketio.emit('safety_alert', {
                        'active': True,
                        'timestamp': notification_timestamp.isoformat(),
                        'count': notification_count,
                        'message': f"{message} (Zone {i})",
                        'screenshot': f"screenshots/{screenshot_name}"
                    })
            zone_last_count[:] = zone_current_count[:]
            zone_current_count[:] = [0] * len(zone_list)

    ppe_stats["date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return frame, zone_frame_arr, roi_has_unsafe

def generate_frames(mode="ppe"):
    if input_type == "image":
        frame = cv2.imread(args.Input)
        processed_frame, _, _ = process_frame(frame, mode)
        display = cv2.resize(processed_frame, (img_rw, img_rh))
        ret, buffer = cv2.imencode('.jpg', display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    cap = cv2.VideoCapture(args.Input if input_type == "video" else camera_index)
    if not cap.isOpened():
        raise RuntimeError('Could not start camera.')
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            if input_type == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        processed_frame, _, _ = process_frame(frame, mode)
        display = cv2.resize(processed_frame, (img_rw, img_rh))
        ret, buffer = cv2.imencode('.jpg', display)
        frame_count += 1
        if frame_count % 5 == 0:
            eventlet.sleep(0.01)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_raw_frames():
    if input_type == "image":
        frame = cv2.imread(args.Input)
        display = cv2.resize(frame, (img_rw, img_rh))
        ret, buffer = cv2.imencode('.jpg', display)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    cap = cv2.VideoCapture(args.Input if input_type == "video" else camera_index)
    if not cap.isOpened():
        raise RuntimeError('Could not start camera.')
    
    frame_count = 0
    while True:
        success, frame = cap.read()
        if not success:
            if input_type == "video":
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            break

        display = cv2.resize(frame, (img_rw, img_rh))
        ret, buffer = cv2.imencode('.jpg', display)
        frame_count += 1
        if frame_count % 5 == 0:
            eventlet.sleep(0.01)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames("ppe"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/danger_zone_feed')
def danger_zone_feed():
    return Response(generate_frames("danger"), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/manage_video_feed')
def manage_video_feed():
    return Response(generate_raw_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/manage')
def manage():
    cap = cv2.VideoCapture(args.Input if input_type in ["video", "image"] else camera_index)
    success, frame = cap.read()
    if success:
        img_oh, img_ow = frame.shape[:2]
        cap.release()
        return render_template('manage.html', img_ow=img_ow, img_oh=img_oh)
    return "Error loading video", 500

@app.route('/save_zone', methods=['POST'])
def save_zone():
    try:
        data = request.json
        coordinates = data.get('coordinates', [])
        if not coordinates:
            return jsonify({"error": "No coordinates provided"}), 400
        
        # Adjust coordinates for original image dimensions
        cap = cv2.VideoCapture(args.Input if input_type in ["video", "image"] else camera_index)
        success, frame = cap.read()
        if not success:
            cap.release()
            return jsonify({"error": "Failed to load frame"}), 500
        
        img_oh, img_ow = frame.shape[:2]
        cap.release()
        x_ratio = img_ow / img_rw
        y_ratio = img_oh / img_rh
        adjusted_coords = [[int(pt[0] * x_ratio), int(pt[1] * y_ratio)] for pt in coordinates]
        
        # Save the zone to the file
        with open(zone_text_path, "a") as f:
            f.write(f"{adjusted_coords}\n")
        
        load_zones()  # Reload zones
        socketio.emit('zone_updated', {'message': 'Zones have been updated'})
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/delete_zone', methods=['POST'])
def delete_zone():
    try:
        data = request.json
        zone_index = data.get('index')
        if zone_index is None:
            return jsonify({"error": "No zone index provided"}), 400
        
        # Read existing zones
        zones = []
        if os.path.exists(zone_text_path):
            with open(zone_text_path, "r") as f:
                zones = [eval(line.strip()) for line in f.readlines() if line.strip()]
        
        # Delete the specified zone
        if 0 <= zone_index < len(zones):
            zones.pop(zone_index)
            with open(zone_text_path, "w") as f:
                for zone in zones:
                    f.write(f"{zone}\n")
            
            load_zones()  # Reload zones
            socketio.emit('zone_updated', {'message': 'Zones have been updated'})
            return jsonify({"status": "success"})
        else:
            return jsonify({"error": "Invalid zone index"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_zones')
def get_zones():
    try:
        zones = []
        if os.path.exists(zone_text_path):
            with open(zone_text_path, "r") as f:
                zones = [eval(line.strip()) for line in f.readlines() if line.strip()]
        
        # Adjust coordinates for display (scale down to canvas size)
        cap = cv2.VideoCapture(args.Input if input_type in ["video", "image"] else camera_index)
        success, frame = cap.read()
        if not success:
            cap.release()
            return jsonify({"zones": []})
        
        img_oh, img_ow = frame.shape[:2]
        cap.release()
        x_ratio = img_ow / img_rw
        y_ratio = img_oh / img_rh
        adjusted_zones = [[[int(pt[0] / x_ratio), int(pt[1] / y_ratio)] for pt in zone] for zone in zones]
        
        return jsonify({"zones": adjusted_zones})
    except:
        return jsonify({"zones": []})

@app.route('/get_initial_frame')
def get_initial_frame():
    cap = cv2.VideoCapture(args.Input if input_type in ["video", "image"] else camera_index)
    success, frame = cap.read()
    if success:
        frame = cv2.resize(frame, (img_rw, img_rh))
        ret, buffer = cv2.imencode('.jpg', frame)
        cap.release()
        return Response(buffer.tobytes(), mimetype='image/jpeg')
    return "Error loading frame", 500

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
    socketio.emit('clear_alerts', {'success': True})
    return jsonify({'success': True})

@app.route('/api/notifications/status')
def notification_status():
    return jsonify({
        'active': notification_active,
        'timestamp': notification_timestamp.isoformat() if notification_timestamp else None,
        'count': notification_count
    })

@app.route('/update_model_settings', methods=['POST'])
def update_model_settings():
    data = request.json
    try:
        ModelSettings.conf_threshold = float(data.get('conf', 0.8))
        ModelSettings.iou_threshold = float(data.get('iou', 0.4))
        return jsonify({"status": "success"})
    except ValueError:
        return jsonify({"error": "Invalid threshold values"}), 400

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('safety_alert', {
        'active': notification_active,
        'timestamp': notification_timestamp.isoformat() if notification_timestamp else None,
        'count': notification_count
    })

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='127.0.0.1', port=5000, debug=True)
