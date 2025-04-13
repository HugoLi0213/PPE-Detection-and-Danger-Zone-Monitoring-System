import argparse
import os
import cv2
import numpy as np
import torch
import datetime
from ultralytics import YOLO
from shapely.geometry import Polygon
from pygrabber.dshow_graph import FilterGraph

# setting
input_type = ""
draw_helmet = 0
draw_vest = 0
text_name_format = "{:s}_{:s}.{:s}"
camera_index = 0
screenshot_name_format = "{:s}-{:%Y%m%d-%H%M%S}.{:s}"

# args setting
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Input", default=None)
args = parser.parse_args()

# check input type
if args.Input is not None:
    # check input exist
    if os.path.exists(args.Input):
        # image
        if args.Input.endswith((".png", ".jpg", ".jpeg")):
            input_type = "image"
            # get text name
            input_name = args.Input.split('\\')[-1]
            text_name = text_name_format.format("zone", input_name, "txt")
        # video
        elif args.Input.endswith(".mp4"):
            input_type = "video"
            input_name = args.Input.split('\\')[-1]
            text_name = text_name_format.format("zone", input_name, "txt")
        # wrong type
        else:
            print("Wrong input format, need to be in [*.png] or [*.jpg] or [*.jpeg] or [*.mp4]")
            exit()
    # input not exist
    else:
        print("input not exist, exiting...")
        exit()
else:
    # check camera exist
    devices = FilterGraph().get_input_devices()
    # camera
    if len(devices) != 0:
        input_type = "camera"
        input_name = devices[camera_index]
        # get text name
        text_name = text_name_format.format("zone", input_name, "txt")
    else:
        print("No Camera is detected, Exiting...")
        exit()

# check dir detected
detected_dir = os.path.join(os.path.dirname(__file__), 'detected')
if not os.path.exists(detected_dir):
    os.makedirs(detected_dir)

# Paths to pre-trained models
models_dir = os.path.join(os.path.dirname(__file__), 'models')
person_model_path = os.path.join(models_dir, 'yolov11s.pt')
ppe_v11_path = os.path.join(models_dir, 'ppe_v11s.pt')
# Check if the model files exist
if not os.path.exists(person_model_path):
    raise FileNotFoundError(f"The person model file '{person_model_path}' does not exist.")
if not os.path.exists(ppe_v11_path):
    raise FileNotFoundError(f"The PPE model file '{ppe_v11_path}' does not exist.")

# Construct the path to the Zone.txt file in the utils directory
zone_text_dir = os.path.join(os.path.dirname(__file__), 'zone')
if not os.path.exists(zone_text_dir):
    os.makedirs(zone_text_dir)
zone_text_path = os.path.join(zone_text_dir, text_name)
# Check if the file exists
if not os.path.exists(zone_text_path):
    print(f"The file '{zone_text_path}' does not exist.")
    open(zone_text_path, "w")
    print(f"The file '{zone_text_path}' is now created, but there is no zone.")
# Open the file
with open(zone_text_path, "r") as f:
    zone_list = f.readlines()
    for z in range(len(zone_list)):
        zone_list[z] = eval(zone_list[z])
    haveZone = True if len(zone_list)>0 else False
    zone_current_count = [0]*len(zone_list)
    zone_last_count = [0]*len(zone_list)

# Initialize YOLO models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
person_model = YOLO(person_model_path).to(device)
ppe_model = YOLO(ppe_v11_path).to(device)

# overlapping for PPE
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
    if object_area == 0:
        return 0
    overlap_ratio = intersection_area / object_area
    return overlap_ratio

# overlapping for zone
def ZONE_overlap(person_box, zone_coord):
    px1, py1, px2, py2 = person_box
    person = Polygon([(px1,py1),(px2,py1),(px2,py2),(px1,py2)])
    for x in range(len(zone_coord)):
        zone_coord[x] = tuple(zone_coord[x])
    zone = Polygon(zone_coord)
    if person.intersects(zone):
        overlap_ratio = person.intersection(zone).area/person.area
        return overlap_ratio
    else:
        return 0

# process frame
def process_frame(frame):
    # frane arr for each zone
    zone_frame_arr = [frame.copy()] * len(zone_list)

    # overlapping setting
    PPE_overlap_threshold = 0.4
    ZONE_overlap_threshold = 0.5

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


    # person
    for person_bbox, person_score in zip(person_bboxes, person_scores):
        wearing_helmet = False
        wearing_vest = False
        # check helmet
        for helmet_bbox in helmet_bboxes:
            overlap_ratio = PPE_overlap(helmet_bbox, person_bbox)
            if overlap_ratio > PPE_overlap_threshold:
                wearing_helmet = True
                break
        # check vest
        for vest_bbox in vest_bboxes:
            overlap_ratio = PPE_overlap(vest_bbox, person_bbox)
            if overlap_ratio > PPE_overlap_threshold:
                wearing_vest = True
                break
        # check in zone
        if not (wearing_helmet and wearing_vest):
            if haveZone:
                for i in range(len(zone_list)):
                    overlap_ratio = ZONE_overlap(person_bbox, zone_list[i])
                    if overlap_ratio > ZONE_overlap_threshold:
                        (px1, py1, px2, py2) = person_bbox
                        cv2.rectangle(zone_frame_arr[i], (px1, py1), (px2, py2), (0, 0, 255), 2)
                        cv2.putText(zone_frame_arr[i], "No PPE", (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                        zone_current_count[i] += 1

        # draw person box
        color = (0, 255, 0) if (wearing_helmet and wearing_vest) else (0, 0, 255)
        (px1, py1, px2, py2) = person_bbox
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        label = "PPE" if wearing_helmet else "No PPE"
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 2)

    # draw helmet box
    if draw_helmet == 1:
        for helmet_bbox in helmet_bboxes:
            (hx1, hy1, hx2, hy2) = helmet_bbox
            cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)
            cv2.putText(frame, "Helmet", (hx1, hy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)

    # draw vest box
    if draw_vest == 1:
        for vest_bbox in vest_bboxes:
            (vx1, vy1, vx2, vy2) = vest_bbox
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), (255, 255, 0), 2)
            cv2.putText(frame, "Vest", (vx1, vy1 - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 2)

    # draw detected person in frame for each zone
    if haveZone:
        for i in range(len(zone_list)):
            zone_color = (255, 255, 255) if (zone_current_count[i] == 0) else (255, 0, 255)
            cv2.polylines(frame,[np.array(zone_list[i])], True, zone_color, 2)
            cv2.polylines(zone_frame_arr[i], [np.array(zone_list[i])], True, zone_color, 2)

    # return processed main frame and processed zone frame
    return frame, zone_frame_arr

def main():
    # image
    if input_type == "image":
        frame = cv2.imread(args.Input)
        frame, zone_frame_arr = process_frame(frame)
        if haveZone:
            notification_status=False
            for x in range(len(zone_current_count)):
                # screenshot if and only if detected count increased
                if zone_current_count[x] > zone_last_count[x]:

                    # notification status
                    notification_status = True

                    # check dir for screenshot
                    input_name_string = input_name.split(".")[0]
                    screenshot_dir = os.path.join(detected_dir, input_name_string)
                    if not os.path.exists(screenshot_dir):
                        os.makedirs(screenshot_dir)

                    # save screenshot(zone frame) if detected no PPE person
                    zone_frame_arr[x] = cv2.resize(zone_frame_arr[x],(1200, 720))
                    screenshot_name = screenshot_name_format.format("zone" + str(x),datetime.datetime.now(),".jpg")
                    path = os.path.join(screenshot_dir, screenshot_name)
                    cv2.imwrite(path, zone_frame_arr[x])

                # save new detected count
                zone_last_count[x] = zone_current_count[x]
                zone_current_count[x] = 0

                # notification
                if notification_status:
                    pass
                    # create a notification for no PPE person detected

        # display processed main frame
        display = cv2.resize(frame, (1200, 720))
        cv2.imshow("frame", display)

    # video
    elif input_type == "video":
        cap = cv2.VideoCapture(args.Input)
        while True:
            # press q to exit the program
            if cv2.waitKey(100) & 0xFF == ord("q"):
                exit()
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from video. Exiting...")
                exit()
            frame, zone_frame_arr = process_frame(frame)
            if haveZone:
                notification_status = False
                for x in range(len(zone_current_count)):
                    # screenshot if and only if detected count increased
                    if zone_current_count[x] > zone_last_count[x]:

                        # notification status
                        notification_status = True

                        # check dir for screenshot
                        input_name_string = input_name.split(".")[0]
                        screenshot_dir = os.path.join(detected_dir, input_name_string)
                        if not os.path.exists(screenshot_dir):
                            os.makedirs(screenshot_dir)

                        # save screenshot(zone frame) if detected no PPE person
                        zone_frame_arr[x] = cv2.resize(zone_frame_arr[x], (1200, 720))
                        screenshot_name = screenshot_name_format.format("zone" + str(x),datetime.datetime.now(),".jpg")
                        path = os.path.join(screenshot_dir, screenshot_name)
                        cv2.imwrite(path, zone_frame_arr[x])

                    # save new detected count
                    zone_last_count[x] = zone_current_count[x]
                    zone_current_count[x] = 0

                    # notification
                    if notification_status:
                        pass
                        # create a notification for no PPE person detected

            # display processed main frame
            display = cv2.resize(frame, (1200, 720))
            cv2.imshow("frame", display)
    # camera
    elif input_type == "camera":
        cap = cv2.VideoCapture(camera_index)
        while(True):
            # press q to exit the program
            if cv2.waitKey(100) & 0xFF == ord("q"):
                exit()
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera. Exiting...")
                exit()
            frame, zone_frame_arr = process_frame(frame)
            if haveZone:
                notification_status = False
                for x in range(len(zone_current_count)):
                    # screenshot if and only if detected count increased
                    if zone_current_count[x] > zone_last_count[x]:

                        # notification status
                        notification_status = True

                        # check dir for screenshot
                        input_name_string = input_name.split(".")[0]
                        screenshot_dir = os.path.join(detected_dir, input_name_string)
                        if not os.path.exists(screenshot_dir):
                            os.makedirs(screenshot_dir)

                        # save screenshot(zone frame) if detected no PPE person
                        zone_frame_arr[x] = cv2.resize(zone_frame_arr[x], (1200, 720))
                        screenshot_name = screenshot_name_format.format("zone" + str(x), datetime.datetime.now(),".jpg")
                        path = os.path.join(screenshot_dir, screenshot_name)
                        cv2.imwrite(path, zone_frame_arr[x])

                    # save new detected count
                    zone_last_count[x] = zone_current_count[x]
                    zone_current_count[x] = 0

                    # notification
                    if notification_status:
                        pass
                        # create a notification for no PPE person detected

            # display processed main frame
            display = cv2.resize(frame, (1200, 720))
            cv2.imshow("frame", display)
    else:
        print("unknown error, exiting...")
        exit()
    if cv2.waitKey(0) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

main()
