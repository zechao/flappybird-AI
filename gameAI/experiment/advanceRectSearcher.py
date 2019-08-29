import time
import cv2
import numpy as np
import gameAI.discretization.geom2D as g2d
import collections


def nothing(x):
    pass


# create window for the slidebars
barsWindow = 'slider'
cv2.namedWindow(barsWindow, cv2.WINDOW_KEEPRATIO)

rl = 'Pipe R Low'
rh = 'Pipe R High'
gl = 'Pipe G Low'
gh = 'Pipe G High'
bl = 'Pipe B Low'
bh = 'Pipe B High'

birdRl = 'Bird R Low'
birdRh = 'Bird R High'
birdGl = 'Bird G Low'
birdGh = 'Bird G High'
birdBl = 'Bird B Low'
birdBh = 'Bird B High'


def createRGBTrackerBar():
    # create the sliders
    cv2.createTrackbar(rl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(rh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(gl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(gh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(bl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(bh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(birdRl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(birdRh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(birdGl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(birdGh, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(birdBl, barsWindow, 0, 255, nothing)
    cv2.createTrackbar(birdBh, barsWindow, 0, 255, nothing)


def getRGBValue(*barNames):
    if len(barNames) == 1:
        return cv2.getTrackbarPos(barNames[0], barsWindow)
    rbg = []
    for bar in barNames:
        rbg.append(cv2.getTrackbarPos(bar, barsWindow))
    return rbg


def setDefaultHSVValue():
    cv2.setTrackbarPos(rl, barsWindow, 84)
    cv2.setTrackbarPos(rh, barsWindow, 84)
    cv2.setTrackbarPos(gl, barsWindow, 56)
    cv2.setTrackbarPos(gh, barsWindow, 56)
    cv2.setTrackbarPos(bl, barsWindow, 71)
    cv2.setTrackbarPos(bh, barsWindow, 71)

    cv2.setTrackbarPos(birdRl, barsWindow, 83)
    cv2.setTrackbarPos(birdRh, barsWindow, 83)
    cv2.setTrackbarPos(birdGl, barsWindow, 56)
    cv2.setTrackbarPos(birdGh, barsWindow, 56)
    cv2.setTrackbarPos(birdBl, barsWindow, 70)
    cv2.setTrackbarPos(birdBh, barsWindow, 70)


createRGBTrackerBar()

imgName = 'frame1000.png'
img = cv2.imread(imgName)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
setDefaultHSVValue()


def searchRect(raster, x, y, rows, cols,allDirection=False):
    minX = rows
    minY = cols
    maxX = x
    maxY = y
    queue = collections.deque()
    queue.append((x, y))
    while len(queue) > 0:
        currentX, currentY = queue.popleft()
        raster[currentX, currentY] = 0
        # right

        # left
        left = currentY - 1
        if left > 0:
            if raster[currentX, left] and raster[currentX, left] != 2:
                queue.append((currentX, left))
                raster[currentX, left] = 2

            # top
        top = currentX - 1
        if top > 0:
            if raster[top, currentY] and raster[top, currentY] != 2:
                queue.append((top, currentY))
                raster[top, currentY] = 2

        right = currentY + 1
        if right < cols:
            if raster[currentX, right] and raster[currentX, right] != 2:
                queue.append((currentX, right))
                raster[currentX, right] = 2
            # down


        down = currentX + 1
        if down < rows:
            if raster[down, currentY] and raster[down, currentY] != 2:
                queue.append((down, currentY))
                raster[down, currentY] = 2


        if allDirection:
            if left > 0 and top > 0:
                if raster[top, left] and raster[top, left] != 2:
                    queue.append((top, left))
                    raster[top, left] = 2

            if right < cols and top > 0:
                if raster[top, right] and raster[top, right] != 2:
                    queue.append((top, right))
                    raster[top, right] = 2

            if right < cols and down < rows:
                if raster[down, right] and raster[down, right] != 2:
                    queue.append((down, right))
                    raster[down, right] = 2

            if left > 0 and down < rows:
                if raster[down, left] and raster[down, left] != 2:
                    queue.append((down, left))
                    raster[down, left] = 2

        minX = min(minX, currentX)
        minY = min(minY, currentY)
        maxX = max(maxX, currentX)
        maxY = max(maxY, currentY)

    return [(minX, minY), (maxX, maxY)]


def fixRect(pipeRects):
    #select all rect with startPoint y = 0
    candidate = []
    result = []
    for rect in pipeRects :
        if rect[0][1] ==0 and rect[1][1]<53 and rect[1][0]-rect[0][0]<=2:
            candidate.append(rect)
        else:
            result.append(rect)
    if len(candidate):
        minX = candidate[0][0][0]
        minY = candidate[0][0][1]
        maxY = candidate[0][1][1]
        for start,end in candidate[1:]:
            minX = min(minX,start[0])
            minY = min(minY,start[1])
            maxY = max(maxY,end[1])
        for i,rect in enumerate(result):
            if abs(minX -rect[0][0])<=3 and abs(rect[0][1] - maxY)<=37:
                result[i]=[(rect[0][0],minY),rect[1]]

    return result


def findRect(binaryImage, minLength=0, transpose=True, heightStart=0, heightEnd=None, widthStart=0, widthEnd=None, allDirection=False):
    """
    :param binaryImage: the edge of image, must contain line of one pixel
    :param minLength: the minim length of line allowed
    :param transpose: transpose the result, to adapt the cv
    :param heightStart:  all pixel with height lower than heightStart it's ignored for check
    :param heightEnd: all pixel with height higher than heightEnd it's ignored for check
    :param widthStart:  all pixel with height lower than widthStart it's ignored for check
    :param widthEnd: all pixel with width higher than widthEnd it's ignored for check
    :return: list of list of points e.g. [[(0,0),(10,0)]]
    """
    rows = binaryImage.shape[0]
    cols = binaryImage.shape[1]
    if heightEnd is not None:
        rows = heightEnd
    if widthEnd is not None:
        cols = widthEnd
    rects = []
    raster = np.copy(binaryImage)

    for i in range(heightStart, rows):
        for j in range(widthStart, cols):
            if raster[i, j]:
                rects.append(searchRect(raster, i, j, rows, cols,allDirection))

    if transpose:
        tLines = []
        for start, end in rects:
            tLines.append([(start[1], start[0]), (end[1], end[0])])
        return tLines
    else:
        return rects


while True:
    cv2.waitKey(int(1000 / 60))
    img = cv2.imread(imgName)  # convert to imgData space experiment

    # get value from bars
    pipe_lower_rgb = np.array(getRGBValue(rl, gl, bl))
    pipe_higher_rgb = np.array(getRGBValue(rh, gh, bh))

    # get value from bars
    bird_lower_rgb = np.array(getRGBValue(birdRl, birdGl, birdBl))
    bird_higher_rgb = np.array(getRGBValue(birdRh, birdGh, birdBh))

    # hard code of pipe
    pipe_lower_rgb = np.array(pipe_lower_rgb)
    pipe_higher_rgb = np.array(pipe_higher_rgb)

    # hard code of bird
    bird_lower_rgb = np.array(bird_lower_rgb)
    bird_higher_rgb = np.array(bird_higher_rgb)

    # build pipe mask
    pipeEdge = cv2.inRange(rgb, pipe_lower_rgb, pipe_higher_rgb)
    cv2.imshow('Pipe mask', pipeEdge)

    pipeRects = findRect(pipeEdge)

    resultUnfix = np.zeros((img.shape[0], img.shape[1], 3), np.float)
    for start, end in pipeRects:
        cv2.rectangle(resultUnfix, start, end, (255, 125, 0), 1)

    cv2.imshow('find Rect with all', resultUnfix)


    pipeEdge[404:406, ] = 0
    # build bird mask
    birdEdge = cv2.inRange(rgb, bird_lower_rgb, bird_higher_rgb)

    result = np.zeros((img.shape[0], img.shape[1], 3), np.float)
    startTime = time.time()

    pipeRects = findRect(pipeEdge, heightEnd=404)

    birdRects = findRect(birdEdge, heightEnd=410, widthStart=50, widthEnd=100,allDirection=True)

    for start, end in pipeRects:
        cv2.rectangle(result, start, end, (255, 125, 0), 1)

    for start, end in birdRects:
        cv2.rectangle(result, start, end, (255, 125, 0), 1)

    fixRectResult = np.zeros((img.shape[0], img.shape[1], 3), np.float)

    pipeRects=fixRect(pipeRects)
    for start, end in pipeRects:
        cv2.rectangle(fixRectResult, start, end, (255, 125, 0), 1)

    print("time speed", time.time() - startTime)
    cv2.imshow('only pipe', pipeEdge)
    cv2.imshow('findRect', result)
    cv2.imshow('image', img)
    cv2.imshow('fixRect',fixRectResult)
cv2.destroyAllWindows()
