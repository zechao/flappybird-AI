import cv2
import numpy as np
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

imgName = 'frame.png'
img = cv2.imread(imgName)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
setDefaultHSVValue()


def isProcessed(lines, p):
    for start, end in lines:
        if start[0] <= p[0] and p[0] <= end[0] and p[1] == start[1] or\
                start[1] <= p[1] and p[1] <= end[1] and start[0]==p[0]:
            return True
    return False


def computeLine(edges, minLength=5, maxSeparation=0,transpose=True):
    img = edges.copy()
    rows = img.shape[0]
    cols = img.shape[1]
    lines = []

    for x in range(0, rows):
        for y in range(0, cols):
            # check if the current point it's already included in other line
            if img[x, y] == 255 and not isProcessed(lines, [x, y]):
                startPoint = (x, y)
                xInc = x + 1
                # search for horizontal line
                while xInc < rows and img[xInc, y] > 0:
                    xInc += 1
                if np.abs(xInc - x) >= minLength:
                    lines.append([startPoint, (xInc, y)])

                 # search for vertical line
                yInc = y + 1
                while yInc < cols and img[x, yInc] > 0:
                    yInc += 1
                if np.abs(yInc - y) >= minLength:
                    lines.append([startPoint, (x, yInc)])

    if transpose:
        tLines = []
        for start, end in lines:
            tLines.append([(start[1],start[0]), (end[1], end[0])])
        return tLines
    else:
        return lines


while True:
    cv2.waitKey(int(1000 / 60))
    img = cv2.imread(imgName) # convert to hsv space experiment
    # get value from bars
    lower_rgb = np.array(getHSVValue(rl, gl, bl))
    higher_rgb = np.array(getHSVValue(rh, gh, bh))

    # hard code of bird
    lower_rgb = np.array(lower_rgb)
    higher_rgb = np.array(higher_rgb)

    # build mask
    edges = cv2.inRange(rgb, lower_rgb, higher_rgb)

    # in order to get line with 1 pixel thickness
    kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(edges, kernel, iterations=1)

    lines = computeLine(erosion)
    for start, end in lines:
        cv2.line(img, start, end, (255, 0, 0), 2)

    cv2.imshow('erosion', erosion)
    cv2.imshow('mask', edges)
    cv2.imshow('experiment', img)
cv2.destroyAllWindows()
