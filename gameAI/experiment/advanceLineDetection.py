import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint


def nothing(x):
    pass


# create window for the slidebars
barsWindow = 'mask'
cv2.namedWindow(barsWindow)

rl = 'R Low'
rh = 'R High'
gl = 'G Low'
gh = 'G High'
bl = 'B Low'
bh = 'B High'


def createHSVTrackBars():
    # create the sliders
    cv2.createTrackbar(rl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(rh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(gl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(gh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(bl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(bh, barsWindow, 0, 255, nothing)


def getHSVValue(*barNames):
    if len(barNames) == 1:
        return cv2.getTrackbarPos(barNames[0], barsWindow)
    hsv = []
    for bar in barNames:
        hsv.append(cv2.getTrackbarPos(bar, barsWindow))
    return hsv


def setDefaultHSVValue():
    cv2.setTrackbarPos(rl, barsWindow, 80)
    cv2.setTrackbarPos(rh, barsWindow, 85)
    cv2.setTrackbarPos(gl, barsWindow, 55)
    cv2.setTrackbarPos(gh, barsWindow, 65)
    cv2.setTrackbarPos(bl, barsWindow, 64)
    cv2.setTrackbarPos(bh, barsWindow, 75)


createHSVTrackBars()

x1 = 0
y1 = 0

img = cv2.imread('day2.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
setDefaultHSVValue()
while True:
    cv2.waitKey(int(1000 / 60))
    img = cv2.imread('day2.png')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to hsv space experiment
    # get value from bars
    lower_rgb = np.array(getHSVValue(rl, gl, bl))
    higher_rgb = np.array(getHSVValue(rh, gh, bh))

    # hard code of bird
    lower_rgb = np.array(lower_rgb)
    higher_rgb = np.array(higher_rgb)

    # build mask
    edges = cv2.inRange(hsv, lower_rgb, higher_rgb)
    # mask = cv2.bitwise_not(mask)
    rho =1  # distance resolution in pixels of the Hough grid
    theta = 1  # angular resolution in radians of the Hough grid
    threshold = 0  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 0  # minimum number of pixels making up a line
    max_line_gap = 0  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines =  cv2.HoughLines(edges, 1, np.pi/120, 70, None,2, 0)

    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(line_image, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow('line_image', line_image)
    cv2.imshow('mask', edges)
    cv2.imshow('experiment', img)
cv2.destroyAllWindows()
