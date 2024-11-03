from flask import Flask, render_template, Response, jsonify, request
from flask_mqtt import Mqtt
import cv2
from ultralytics import YOLO
from utils.detector import PPEDetector
from utils.zone_monitor import ZoneMonitor

app = Flask(__name__)

# MQTT Configuration
app.config['MQTT_BROKER_URL'] = 'broker.emqx.io'  # Use your broker URL
app.config['MQTT_BROKER_PORT'] = 1883  # Default MQTT port
app.config['MQTT_USERNAME'] = ''  # Set if your broker requires authentication
app.config['MQTT_PASSWORD'] = ''  # Set if your broker requires authentication
app.config['MQTT_KEEPALIVE'] = 5  # KeepAlive time in seconds
app.config['MQTT_TLS_ENABLED'] = False  # Set to True if your broker supports TLS

mqtt_client = Mqtt(app)

# Initialize detectors
ppe_detector = PPEDetector(model_path='models/best.pt')
zone_monitor = ZoneMonitor()

# Initialize video capture (replace with your CCTV stream URL)
camera = cv2.VideoCapture(0)

def generate_ppe_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Perform PPE detection
        frame, violations = ppe_detector.detect_ppe(frame)
        
        # Publish violations to MQTT
        for violation in violations:
            mqtt_client.publish('/alerts/ppe', str(violation))
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def generate_zone_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Perform zone monitoring
        frame, violations = zone_monitor.monitor_zones(frame)
        
        # Publish violations to MQTT
        for violation in violations:
            mqtt_client.publish('/alerts/zone', str(violation))
        
        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/manage')
def manage():
    return render_template('manage.html')

@app.route('/stats')
def stats():
    return render_template('stats.html')

@app.route('/video_feed_ppe')
def video_feed_ppe():
    return Response(generate_ppe_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_zone')
def video_feed_zone():
    return Response(generate_zone_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/publish', methods=['POST'])
def publish_message():
    request_data = request.get_json()
    publish_result = mqtt_client.publish(request_data['topic'], request_data['msg'])
    return jsonify({'code': publish_result[0]})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
