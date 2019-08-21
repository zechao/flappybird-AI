import cv2
import numpy as np


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


def getTrackBarValues(*barNames):
    if len(barNames) == 1:
        return cv2.getTrackbarPos(barNames[0], barsWindow)
    hsv = []
    for bar in barNames:
        hsv.append(cv2.getTrackbarPos(bar, barsWindow))
    return hsv


def setDefaultHSVValue():
    cv2.setTrackbarPos(hl, barsWindow, get_bird_hsv_low()[0])
    cv2.setTrackbarPos(hh, barsWindow, get_bird_hsv_high()[0])
    cv2.setTrackbarPos(sl, barsWindow, get_bird_hsv_low()[1])
    cv2.setTrackbarPos(sh, barsWindow, get_bird_hsv_high()[1])
    cv2.setTrackbarPos(vl, barsWindow, get_bird_hsv_low()[2])
    cv2.setTrackbarPos(vh, barsWindow, get_bird_hsv_high()[2])


def get_bird_hsv_low():
    return [0, 237, 204]


def get_bird_hsv_high():
    return [170, 255, 255]


createHSVTrackBars()

x1 = 0
y1 = 0

img = cv2.imread('img.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
setDefaultHSVValue()
while True:
    cv2.waitKey(int(1000 / 60))
    img = cv2.imread('img.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to imgData space experiment

    # get value from bars
    lower_hsv = np.array(getTrackBarValues(hl, sl, vl))
    higher_hsv = np.array(getTrackBarValues(hh, sh, vh))

    # range color of bird body in order to track the bird
    lower_hsv = np.array(lower_hsv)
    higher_hsv = np.array(higher_hsv)

    # build mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    all_cnt_img = cv2.drawContours(np.copy(img), contours, -1, (0, 255, 0), 3)
    if len(contours) != 0:
        cnt = contours[0]
        area = cv2.contourArea(cnt)



        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

    cv2.imshow('pipeContours', all_cnt_img)
    cv2.imshow('mask', mask)
    cv2.imshow('experiment', img)
cv2.destroyAllWindows()
