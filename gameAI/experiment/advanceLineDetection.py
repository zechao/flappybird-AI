import time
import cv2
import numpy as np
import gameAI.discretization.geom2D as g2d


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
    cv2.setTrackbarPos(rh, barsWindow, 85)
    cv2.setTrackbarPos(gl, barsWindow, 56)
    cv2.setTrackbarPos(gh, barsWindow, 58)
    cv2.setTrackbarPos(bl, barsWindow, 71)
    cv2.setTrackbarPos(bh, barsWindow, 73)

    cv2.setTrackbarPos(birdRl, barsWindow, 83)
    cv2.setTrackbarPos(birdRh, barsWindow, 83)
    cv2.setTrackbarPos(birdGl, barsWindow, 56)
    cv2.setTrackbarPos(birdGh, barsWindow, 56)
    cv2.setTrackbarPos(birdBl, barsWindow, 70)
    cv2.setTrackbarPos(birdBh, barsWindow, 70)


createRGBTrackerBar()

imgName = 'frame.png'
img = cv2.imread(imgName)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
setDefaultHSVValue()


def isProcessed(lines, x,y):
    for start, end in lines:
        vertical = start[0] <= x and x <= end[0] and y== start[1]
        horizontal = start[1] <= y and y <= end[1] and start[0] == x
        if vertical or horizontal:
            return True
    return False


def findLines(edges, minLength=0, transpose=True, heightStart=0, heightEnd=None, widthStart=0, widthEnd=None):
    """
    :param edges: the edge of image, must contain line of one pixel
    :param minLength: the minim length of line allowed
    :param transpose: transpose the result, to adapt the cv
    :param heightStart:  all pixel with height lower than heightStart it's ignored for check
    :param heightEnd: all pixel with height higher than heightEnd it's ignored for check
    :param widthStart:  all pixel with height lower than widthStart it's ignored for check
    :param widthEnd: all pixel with width higher than widthEnd it's ignored for check
    :return: list of list of points e.g. [[(0,0),(10,0)]]
    """
    rows = edges.shape[0]
    cols = edges.shape[1]
    if heightEnd is not None:
        rows = heightEnd
    if widthEnd is not None:
        cols = widthEnd
    lines = []

    for x in range(heightStart, rows):
        for y in range(widthStart, cols):
            # check if the current point it's already included in other line
            if edges[x, y] == 255 and not isProcessed(lines, x, y):
                startPoint = (x, y)
                xInc = x
                # search for horizontal line
                while xInc + 1 < rows and edges[xInc + 1, y] > 0:
                    xInc += 1
                if np.abs(xInc - x) >= minLength:
                    lines.append([startPoint, (xInc, y)])

                # search for vertical line
                yInc = y
                while yInc + 1 < cols and edges[x, yInc + 1] > 0:
                    yInc += 1
                if np.abs(yInc - y) >= minLength:
                    lines.append([startPoint, (x, yInc)])
    if transpose:
        tLines = []
        for start, end in lines:
            tLines.append([(start[1], start[0]), (end[1], end[0])])
        return tLines
    else:
        return lines


def fixDisconnectedLines(lines, maxDisconnection):
    if maxDisconnection == 0:
        return lines
    linesCopy = lines[:]
    for i, line1 in enumerate(linesCopy):
        # search pipeLines with same x position
        for j, line2 in enumerate(linesCopy):
            if line1 != line2 and line1[1][0] == line2[1][0] == line1[0][0] == line2[0][0]:
                distance = np.abs(line1[1][1] - line2[0][1])
                if distance <= maxDisconnection:
                    linesCopy[i] = [line1[0], line2[1]]
                    linesCopy.remove(linesCopy[j])
    return linesCopy


def findRect(birdLines):
    """
    from an list of lines find the rect which include all lines
    :param birdLines:
    :return:
    """
    if len(birdLines) == 0:
        return None
    if len(birdLines) == 1:
        return birdLines[0]
    # start point must be minus
    x1 = birdLines[0][0][0]
    y1 = birdLines[0][0][1]
    # end point must be bigger
    x2 = birdLines[0][1][0]
    y2 = birdLines[0][1][1]
    for line in birdLines[1:]:
        if line[0][0] <= x1:
            x1 = line[0][0]
        if line[0][1] <= y1:
            y1 = line[0][1]
        if line[1][0] >= x2:
            x2 = line[1][0]
        if line[1][1] >= y2:
            y2 = line[1][1]

    return [(x1 - 3, y1 - 3), (x2 + 3, y2 + 3)]


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

    # build bird mask
    birdEdge = cv2.inRange(rgb, bird_lower_rgb, bird_higher_rgb)

    # in order to get line with 1 pixel thickness
    kernel = np.ones((2, 2), np.uint8)
    pipeErosion = cv2.erode(pipeEdge, kernel, iterations=1)

    # in order to get line with 1 pixel thickness
    kernel = np.ones((2, 2), np.uint8)
    birdErosion = cv2.erode(birdEdge, kernel, iterations=1)
    start =time.time()
    pipeLines = findLines(pipeErosion, heightEnd=410)
    print("time spent:", time.time()-start)
    birdLines = findLines(birdErosion, heightEnd=410, widthStart=50, widthEnd=100)
    birdRect = findRect(birdLines)

    pipeLines = fixDisconnectedLines(pipeLines, 40)

    for start, end in pipeLines:
        cv2.line(img, start, end, (255, 0, 0), 2)

    cv2.rectangle(img, birdRect[0], birdRect[1], [255] * 3)

    cv2.imshow('pipeErosion', pipeErosion)
    cv2.imshow('birdErosion', birdErosion)
    cv2.imshow('mask', pipeEdge)
    cv2.imshow('findLines', img)
cv2.destroyAllWindows()
