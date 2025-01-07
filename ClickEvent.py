import cv2


def mouse(event, x, y, flags, param): # click event to print coordination
    if event == cv2.EVENT_LBUTTONDOWN:
        print(int(x*(img_ow/img_rw)), int(y*(img_oh/img_rh)))

img = cv2.imread("pic2.jpg")
img_oh = img.shape[0] # original height
img_ow = img.shape[1] # original width
img_rh = 720 # resize height
img_rw = 1200 # resize width

frame = cv2.resize(img, (img_rw,img_rh))
cv2.imshow("frame", frame)


cv2.setMouseCallback('frame', mouse)
cv2.waitKey(0)
cv2.destroyAllWindows()