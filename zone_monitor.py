import cv2
import numpy as np
from ultralytics import YOLO


class ZoneMonitor:
    def __init__(self, image_path, model_path, zone_box):
        self.image_path = image_path
        self.model_path = model_path
        self.zone_box = np.array(zone_box, dtype=int)
        self.frame = None
        self.model = None
        self.overlap_threshold = 0.5

    def load_image(self):
        try:
            self.frame = cv2.imread(self.image_path)
            if self.frame is None:
                raise FileNotFoundError(f"Image not found: {self.image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

    def initialize_model(self):
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise

    def calculate_overlap(self, object_box, area_box):
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

    def monitor_zone(self):
        try:
            results = self.model(self.frame, device="cpu")
            result = results[0]
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            scores = np.array(result.boxes.conf.cpu(), dtype="float")

            for bbox, score in zip(bboxes, scores):
                overlap_ratio = self.calculate_overlap(bbox, self.zone_box)
                if overlap_ratio > self.overlap_threshold:
                    (x1, y1, x2, y2) = bbox
                    cv2.rectangle(self.frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(self.frame, "In Zone", (x1, y1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                    print(f"Object detected in zone with score: {score}")

            zx1, zy1, zx2, zy2 = self.zone_box
            cv2.rectangle(self.frame, (zx1, zy1), (zx2, zy2), (0, 255, 255), 2)
            cv2.putText(self.frame, "Zone", (zx1, zy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

        except Exception as e:
            print(f"Error during zone monitoring: {e}")
            raise

    def display_results(self):
        try:
            resized_frame = cv2.resize(self.frame, (1200, 720))
            cv2.imshow("Zone Monitoring", resized_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying results: {e}")
            raise

if __name__ == "__main__":
    image_path = "utils/pic1.jpg"  # Ensure this path is correct
    model_path = "models/yolov8s.pt"  # Ensure this path is correct
    zone_box = [291, 331, 2354, 1353]  # Define your zone box coordinates

    zone_monitor = ZoneMonitor(image_path, model_path, zone_box)
    zone_monitor.load_image()
    zone_monitor.initialize_model()
    zone_monitor.monitor_zone()
    zone_monitor.display_results()
