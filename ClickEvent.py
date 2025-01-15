import cv2


def mouse(event, x, y, flags, param): # click event to print coordination
    # left button: record coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        temp_x=int(x*(img_ow/img_rw))
        temp_y=int(y*(img_oh/img_rh))
        print(temp_x,temp_y)
        f=open("ROI_coord.txt","a")
        string=str(temp_x)+" "+str(temp_y)+" "
        f.write(string)
        f.close()
    # right button: clear
    if event == cv2.EVENT_RBUTTONDOWN:
        print("cleared")
        f=open("ROI_coord.txt","w")
        f.write("")
        f.close()
    # middle button: display current coordinates
    if event == cv2.EVENT_MBUTTONDOWN:
        f=open("ROI_coord.txt","r")
        coord = f.read().split()
        print(coord)

cap = cv2.VideoCapture("BM02.mp4")
ret, img = cap.read()
img_oh = img.shape[0] # original height
img_ow = img.shape[1] # original width
img_rh = 720 # resize height
img_rw = 1200 # resize width

frame = cv2.resize(img, (img_rw,img_rh))
cv2.imshow("frame", frame)

cv2.setMouseCallback('frame', mouse)
cv2.waitKey(0)
# press q to quit
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()