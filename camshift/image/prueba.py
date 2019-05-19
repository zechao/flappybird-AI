import cv2
import numpy as np
from matplotlib import pyplot as plt


def nothing(x):
    pass
# create window for the slidebars
barsWindow = 'mask'
cv2.namedWindow(barsWindow)

hl = 'H Low'
hh = 'H High'
sl = 'S Low'
sh = 'S High'
vl = 'V Low'
vh = 'V High'

def createHSVTrackBars():
    # create the sliders
    cv2.createTrackbar(hl, barsWindow, 0, 179, nothing)
    cv2.createTrackbar(hh, barsWindow, 0, 179, nothing)
    cv2.createTrackbar(sl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(sh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(vl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(vh, barsWindow, 0, 255, nothing)


def getHSVValue(*barNames):
    if len(barNames)== 1:
        return cv2.getTrackbarPos(barNames[0],barsWindow)
    hsv =[]
    for bar in barNames:
        hsv.append(cv2.getTrackbarPos(bar,barsWindow))
    return hsv


createHSVTrackBars()

x1=0
y1=0
def OnMouseAction(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("左键点击")
    elif event==cv2.EVENT_RBUTTONDOWN :
        print("右键点击")
    elif flags==cv2.EVENT_FLAG_LBUTTON:
        print("左鍵拖曳")
        x1 = x
        y1 = y
        print(x1,y1 )
    elif event==cv2.EVENT_MBUTTONDOWN :
        print("中键点击")

pixel =[]

img = cv2.imread('im90.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.show()
while True:
    cv2.waitKey(int(1000/60))
    img= cv2.imread('im90.png')

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_hsv = np.array(getHSVValue(hl, sl, vl))
    higher_hsv = np.array(getHSVValue(hh,sh,vh))
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    frame = cv2.bitwise_and(img, img, mask)
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    all_cnt_img = cv2.drawContours(np.copy(img), contours, -1, (0, 255, 0), 3)

    cv2.setMouseCallback('image',OnMouseAction)
    cv2.imshow('contours', all_cnt_img)
    cv2.imshow('mask', mask)
    cv2.imshow('image', frame)
cv2.destroyAllWindows()
