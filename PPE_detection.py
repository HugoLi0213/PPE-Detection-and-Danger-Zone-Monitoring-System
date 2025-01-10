import cv2
import numpy as np
from ultralytics import YOLO


class PPEChecker:
    def __init__(self, image_path, person_model_path, ppe_model_path):
        self.image_path = image_path
        self.person_model_path = person_model_path
        self.ppe_model_path = ppe_model_path
        self.frame = None
        self.person_model = None
        self.ppe_model = None
        self.ROI_box = np.array([291, 331, 2354, 1353], dtype=int)
        self.ROI_threshold = 0.5
        self.ROI_count = 0
        self.overlap_threshold = 0.4

    def load_image(self):
        try:
            self.frame = cv2.imread(self.image_path)
            if self.frame is None:
                raise FileNotFoundError(f"Image not found: {self.image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            raise

    def initialize_models(self):
        try:
            self.person_model = YOLO(self.person_model_path)
            self.ppe_model = YOLO(self.ppe_model_path)
        except Exception as e:
            print(f"Error initializing models: {e}")
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

    def perform_detection(self):
        try:
            person_results = self.person_model(self.frame, device="cpu")
            person_result = person_results[0]
            person_bboxes = np.array(person_result.boxes.xyxy.cpu(), dtype="int")
            person_classes = np.array(person_result.boxes.cls.cpu(), dtype="int")
            person_scores = np.array(person_result.boxes.conf.cpu(), dtype="float")
            person_indices = np.where(person_classes == 0)[0]
            person_bboxes = person_bboxes[person_indices]
            person_scores = person_scores[person_indices]

            ppe_results = self.ppe_model(self.frame, device="cpu", imgsz=640, conf=0.8, iou=0.4)
            ppe_result = ppe_results[0]
            ppe_bboxes = np.array(ppe_result.boxes.xyxy.cpu(), dtype="int")
            ppe_classes = np.array(ppe_result.boxes.cls.cpu(), dtype="int")
            ppe_scores = np.array(ppe_result.boxes.conf.cpu(), dtype="float")
            helmet_indices = np.where(ppe_classes == 0)[0]
            helmet_bboxes = ppe_bboxes[helmet_indices]
            vest_indices = np.where(ppe_classes == 1)[0]
            vest_bboxes = ppe_bboxes[vest_indices]

            for person_bbox, person_score in zip(person_bboxes, person_scores):
                wearing_helmet = False
                wearing_vest = False

                for helmet_bbox in helmet_bboxes:
                    overlap_ratio = self.calculate_overlap(helmet_bbox, person_bbox)
                    if overlap_ratio > self.overlap_threshold:
                        wearing_helmet = True
                        break

                for vest_bbox in vest_bboxes:
                    overlap_ratio = self.calculate_overlap(vest_bbox, person_bbox)
                    if overlap_ratio > self.overlap_threshold:
                        wearing_vest = True
                        break

                if not (wearing_helmet and wearing_vest):
                    overlap_ratio = self.calculate_overlap(person_bbox, self.ROI_box)
                    if overlap_ratio > self.ROI_threshold:
                        self.ROI_count += 1

                color = (0, 255, 0) if (wearing_helmet and wearing_vest) else (0, 0, 255)
                (px1, py1, px2, py2) = person_bbox
                cv2.rectangle(self.frame, (px1, py1), (px2, py2), color, 2)
                label = "PPE" if wearing_helmet else "No PPE"
                cv2.putText(self.frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

            for helmet_bbox in helmet_bboxes:
                (hx1, hy1, hx2, hy2) = helmet_bbox
                cv2.rectangle(self.frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
                cv2.putText(self.frame, "Helmet", (hx1, hy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

            for vest_bbox in vest_bboxes:
                (vx1, vy1, vx2, vy2) = vest_bbox
                cv2.rectangle(self.frame, (vx1, vy1), (vx2, vy2), (255, 255, 0), 2)
                cv2.putText(self.frame, "Vest", (vx1, vy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

            ROIx1, ROIy1, ROIx2, ROIy2 = self.ROI_box
            cv2.rectangle(self.frame, (ROIx1, ROIy1), (ROIx2, ROIy2), (0, 255, 255), 2)
            cv2.putText(self.frame, "ROI", (ROIx1, ROIy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 2)
            print("ROI count", self.ROI_count)

        except Exception as e:
            print(f"Error during detection: {e}")
            raise

    def display_results(self):
        try:
            framers = cv2.resize(self.frame, (1200, 720))
            cv2.imshow("Image", framers)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error displaying results: {e}")
            raise

if __name__ == "__main__":
    image_path = "pic1.jpg"
    person_model_path = "yolov8s.pt"
    ppe_model_path = "ovu.pt"

    ppe_checker = PPEChecker(image_path, person_model_path, ppe_model_path)
    ppe_checker.load_image()
    ppe_checker.initialize_models()
    ppe_checker.perform_detection()
    ppe_checker.display_results()
