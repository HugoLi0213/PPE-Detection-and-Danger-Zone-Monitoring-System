import os

import cv2
import numpy as np
from ultralytics import YOLO


class PPEDetector:
    def __init__(self):
        # Get current working directory
        self.cwd = os.getcwd()
        
        # Define paths relative to current directory
        self.image_path = os.path.join(self.cwd, "pic3.jpg")
        self.person_model_path = os.path.join(self.cwd, "yolov8s.pt")
        self.ppe_model_path = os.path.join(self.cwd, "ovu.pt")
        
        # Verify file existence
        self._verify_files()
        
        # Initialize models
        self.person_model = YOLO(self.person_model_path)
        self.ppe_model = YOLO(self.ppe_model_path)
        
        # Original thresholds
        self.overlap_threshold = 0.4
        self.roi_threshold = 0.5
        
    def _verify_files(self):
        """Verify that required files exist"""
        files_to_check = [
            (self.image_path, "Image file"),
            (self.person_model_path, "Person detection model"),
            (self.ppe_model_path, "PPE detection model")
        ]
        
        for file_path, file_desc in files_to_check:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_desc} not found at: {file_path}")

    def calculate_overlap(self, object_box, area_box):
        try:
            ox1, oy1, ox2, oy2 = map(float, object_box)
            ax1, ay1, ax2, ay2 = map(float, area_box)
            
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
                
            return float(intersection_area / object_area)
        except Exception as e:
            print(f"Error in overlap calculation: {e}")
            return 0

    def process_frame(self):
        # Read image
        frame = cv2.imread(self.image_path)
        if frame is None:
            raise ValueError(f"Failed to load image from {self.image_path}")

        # Person detection
        person_results = self.person_model(frame, device="cpu")
        person_result = person_results[0]
        person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
        person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
        person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")

        # PPE detection
        ppe_results = self.ppe_model(frame, device="cpu", imgsz=640, conf=0.8, iou=0.4)
        ppe_result = ppe_results[0]
        ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
        ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
        ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")

        # Filter detections
        person_indices = np.where(person_classes == 0)[0]
        person_bboxes = person_bboxes[person_indices]
        person_scores = person_scores[person_indices]

        helmet_indices = np.where(ppe_classes == 0)[0]
        helmet_bboxes = ppe_bboxes[helmet_indices]
        helmet_scores = ppe_scores[helmet_indices]

        vest_indices = np.where(ppe_classes == 1)[0]
        vest_bboxes = ppe_bboxes[vest_indices]
        vest_scores = ppe_scores[vest_indices]

        # ROI setup
        ROI_box = np.array([291, 331, 2354, 1353], dtype=int)
        ROI_count = 0

        # Process detections
        for person_bbox, person_score in zip(person_bboxes, person_scores):
            wearing_helmet = False
            wearing_vest = False

            # Check helmet
            for helmet_bbox in helmet_bboxes:
                overlap = self.calculate_overlap(helmet_bbox, person_bbox)
                if overlap and overlap > self.overlap_threshold:
                    wearing_helmet = True
                    break

            # Check vest
            for vest_bbox in vest_bboxes:
                overlap = self.calculate_overlap(vest_bbox, person_bbox)
                if overlap and overlap > self.overlap_threshold:
                    wearing_vest = True
                    break

            # ROI check
            if not (wearing_helmet and wearing_vest):
                overlap = self.calculate_overlap(person_bbox, ROI_box)
                if overlap and overlap > self.roi_threshold:
                    ROI_count += 1

            # Draw bounding boxes
            color = (0, 255, 0) if (wearing_helmet and wearing_vest) else (0, 0, 255)
            px1, py1, px2, py2 = person_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
            label = "PPE" if (wearing_helmet and wearing_vest) else "No PPE"
            cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

        # Draw ROI
        cv2.rectangle(frame, (ROI_box[0], ROI_box[1]), (ROI_box[2], ROI_box[3]), (0, 255, 255), 2)
        cv2.putText(frame, "ROI", (ROI_box[0], ROI_box[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)

        print(f"ROI count: {ROI_count}")
        
        # Display results
        frame_resized = cv2.resize(frame, (1200, 720))
        cv2.imshow("PPE Detection", frame_resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    try:
        detector = PPEDetector()
        detector.process_frame()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
