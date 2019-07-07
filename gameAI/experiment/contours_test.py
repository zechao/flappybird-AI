import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint


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
    if len(barNames) == 1:
        return cv2.getTrackbarPos(barNames[0], barsWindow)
    hsv = []
    for bar in barNames:
        hsv.append(cv2.getTrackbarPos(bar, barsWindow))
    return hsv


def setDefaultHSVValue():
    cv2.setTrackbarPos(hl, barsWindow, 0)
    cv2.setTrackbarPos(hh, barsWindow, 170)
    cv2.setTrackbarPos(sl, barsWindow, 0)
    cv2.setTrackbarPos(sh, barsWindow, 255)
    cv2.setTrackbarPos(vl, barsWindow, 95)
    cv2.setTrackbarPos(vh, barsWindow, 255)


createHSVTrackBars()

x1 = 0
y1 = 0

img = cv2.imread('img.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
setDefaultHSVValue()
while True:
    cv2.waitKey(int(1000 / 60))
    img = cv2.imread('img.jpg')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # convert to hsv space experiment
    # get value from bars
    lower_hsv = np.array(getHSVValue(hl, sl, vl))
    higher_hsv = np.array(getHSVValue(hh, sh, vh))

    # hard code of bird
    lower_hsv = np.array(lower_hsv)
    higher_hsv = np.array(higher_hsv)

    # build mask
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)
    # frame = cv2.bitwise_and(img, img, mask)
    contours, hierarchy = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE )

    all_cnt_img = cv2.drawContours(np.copy(img), contours, -1, (255, 255, 0), 2)
    final_result = np.copy(img)
    if len(contours) != 0:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 3)
            if x < 80 and w * h < 700:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), cv2.FILLED)
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), cv2.FILLED)

    cv2.imshow('contours', all_cnt_img)
    cv2.imshow('mask', mask)
    cv2.imshow('experiment', img)
cv2.destroyAllWindows()
