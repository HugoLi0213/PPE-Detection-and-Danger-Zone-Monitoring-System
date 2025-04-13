import argparse
import os
import cv2
import numpy as np
from pygrabber.dshow_graph import FilterGraph

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--Input", default=None, help="input a image or video, use camera if no input")  # cmd: Python Zone.py -i [Picture or Video]
parser.add_argument("-m", "--Mode", default=None, help="[0] for draw, [1] for view, [2] for delete")
args = parser.parse_args()

# settings
img_rh = 720  # resized height
img_rw = 1200  # resized width
red = (0, 0, 255)
green = (0, 255, 0)
arr_index = 0
text_name_format = "{:s}_{:s}.{:s}"
waitKeyValue = 0
camera_index = 0

# check zone dir
zone_dir = os.path.join(os.path.dirname(__file__), 'zone')
if not os.path.exists(zone_dir):
    os.makedirs(zone_dir)

def main(image):
    # image resolution
    img_oh = image.shape[0]  # original height
    img_ow = image.shape[1]  # original width
    # resolution ratio
    x_ratio = img_ow / img_rw
    y_ratio = img_oh / img_rh
    # resize
    img = cv2.resize(image, (img_rw, img_rh))
    # Mode - Create
    if args.Mode == "0":
        dot_array = []
        def draw(event, x, y, flags, param):
            # left click - add one dot
            if event == cv2.EVENT_LBUTTONDOWN:
                dot_array.append([x, y])
            # right click - remove one dot
            if event == cv2.EVENT_RBUTTONDOWN:
                if len(dot_array) != 0:
                    dot_array.pop()
            # middle click - save dot to txt
            if event == cv2.EVENT_MBUTTONDOWN:
                temp_arr = []
                for dot in dot_array:
                    temp_arr.append([int(dot[0] * x_ratio), int(dot[1] * y_ratio)])
                if temp_arr != []:
                    f = open(zone_path, "a")
                    f.write(f"{temp_arr}\n")
                    f.close()
                print("Saved")
                exit()
            # show lines
            if len(dot_array) != 0:
                img2 = img.copy()
                for dot in dot_array:
                    cv2.circle(img2, dot, 2, red, -1)
                    if dot_array.index(dot) != 0:
                        cv2.line(img2, last_dot, dot, red, 2)
                    if dot_array.index(dot) == len(dot_array) - 1:
                        cv2.line(img2, dot, dot_array[0], green, 2)
                    last_dot = dot
                    cv2.imshow("DrawZone", img2)
            else:
                cv2.imshow("DrawZone", img)
        cv2.imshow("DrawZone", img)
        cv2.setMouseCallback('DrawZone', draw)
        cv2.waitKey(waitKeyValue)
    # Mode - View
    elif args.Mode == "1":
        if os.path.isfile(zone_path):
            f = open(zone_path, "r")
            zone_list = f.readlines()
            f.close()
            if zone_list is not None:
                for l in range(len(zone_list)):
                    zone_list[l] = eval(zone_list[l])
                    for p in range(len(zone_list[l])):
                        zone_list[l][p] = [int(zone_list[l][p][0] / x_ratio), int(zone_list[l][p][1] / y_ratio)]
                    cv2.polylines(img, [np.array(zone_list[l])], True, (255, 0, 0), 2)
            cv2.imshow("ShowZone", img)
            cv2.waitKey(waitKeyValue)
        else:
            print("No Zone Drew, Exiting...")
            exit()
    # Mode - Delete
    elif args.Mode == "2":
        path = f"./zone/{text_name}"
        if os.path.isfile(path):
            f = open(zone_path, "r")
            zone_list = f.readlines()
            f.close()
            if len(zone_list) == 0:
                print("No Zone Drew, Exiting...")
                exit()
            else:
                for l in range(len(zone_list)):
                    zone_list[l] = eval(zone_list[l])
                    for p in range(len(zone_list[l])):
                        zone_list[l][p] = [int(zone_list[l][p][0] / x_ratio), int(zone_list[l][p][1] / y_ratio)]
                remove_arr = [False] * len(zone_list)
                def remove(event, x, y, flags, param):
                    global arr_index
                    img_copy = img.copy()
                    # left click - next zone
                    if event == cv2.EVENT_LBUTTONDOWN:
                        if arr_index == len(zone_list) - 1:
                            arr_index = 0
                        else:
                            arr_index += 1
                        if remove_arr[arr_index] == False:
                            color = green
                        elif remove_arr[arr_index] == True:
                            color = red
                        cv2.polylines(img_copy, [np.array(zone_list[arr_index])], True, color, 2)
                        cv2.imshow("RemoveZone", img_copy)
                    # right click - remove zone
                    if event == cv2.EVENT_RBUTTONDOWN:
                        if remove_arr[arr_index] == False:
                            remove_arr[arr_index] = True
                            color = red
                        elif remove_arr[arr_index] == True:
                            remove_arr[arr_index] = False
                            color = green
                        cv2.polylines(img_copy, [np.array(zone_list[arr_index])], True, color, 2)
                        cv2.imshow("RemoveZone", img_copy)
                    # middle click - save changes
                    if event == cv2.EVENT_MBUTTONDOWN:
                        for x in range(len(zone_list)):
                            if remove_arr[x] == True:
                                zone_list[x] = None
                        string = ""
                        for zone in zone_list:
                            if zone is not None:
                                temp_arr = []
                                for pt in zone:
                                    temp_arr.append([int(pt[0] * x_ratio), int(pt[1] * y_ratio)])
                                string += f"{temp_arr}\n"
                        f = open(zone_path, "w")
                        f.write(string)
                        f.close()
                        print("Saved")
                        exit()
            img_copy = img.copy()
            cv2.polylines(img_copy, [np.array(zone_list[arr_index])], True, green, 2)
            cv2.imshow("RemoveZone", img_copy)
            cv2.setMouseCallback('RemoveZone', remove)
            cv2.waitKey(waitKeyValue)
        else:
            print("No Zone Drew, Exiting...")
            exit()
    else:
        print("Input a mode, [0] for draw, [1] for view, [2] for delete")
        exit()

# check input type
if args.Input is not None:
    # check input exist
    if os.path.isfile(args.Input):
        # if input wrong type
        if not args.Input.endswith((".png", ".jpg", ".jpeg", ".mp4")):
            print("Wrong input format, need to be in [*.png] or [*.jpg] or [*.jpeg] or [*.mp4], exiting...")
            exit()
        # get text name
        else:
            input_name = args.Input.split('\\')[-1]
            text_name = text_name_format.format("zone", input_name, "txt")
    # input not exist
    else:
        print("file not exist, exiting...")
        exit()
# camera
else:
    # check camera exist
    devices = FilterGraph().get_input_devices()
    if len(devices) != 0:
        input_name = devices[camera_index]
        # get text name
        text_name = text_name_format.format("zone", input_name, "txt")
    # no camera
    else:
        print("No Camera is detected, Exiting...")
        exit()

# check txt file exists if Mode is view and delete
zone_path = os.path.join(zone_dir, text_name)
if args.Mode == ("1" or "2"):
    if not os.path.isfile(zone_path):
        print("No Zone Drew, Exiting...")
        exit()

# check mode
if args.Mode == None:
    print("Input a mode, [0] for draw, [1] for view, [2] for delete")
    exit()

# check type before main()
if args.Input is not None:
    # image
    if args.Input.endswith((".png", ".jpg", ".jpeg")):
        image = cv2.imread(args.Input)
        main(image)
    # video
    elif args.Input.endswith((".mp4")):
        cap = cv2.VideoCapture(args.Input)
        while (True):
            if cv2.waitKey(waitKeyValue) & 0xFF == ord('q'):
                exit()
            ret, image = cap.read()
            if not ret:
                break
            main(image)
else:
    # camera
    cap = cv2.VideoCapture(camera_index)
    while(True):
        if cv2.waitKey(waitKeyValue) & 0xFF == ord('q'):
            exit()
        ret, image = cap.read()
        if not ret:
            break
        main(image)

# press r to refresh
if cv2.waitKey(waitKeyValue) & 0xFF == ord('r'):
    cv2.destroyAllWindows()
