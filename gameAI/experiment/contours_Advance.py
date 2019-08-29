import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
import time
import math

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


createRGBTrackerBar()

imgName = 'frame1000.png'
img = cv2.imread(imgName)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
setDefaultHSVValue()

while True:
    cv2.waitKey(int(1000 / 60))
    img = cv2.imread(imgName)
    imgData = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to imgData space experiment
    # get value from bars

    pipe_lower_rgb = np.array(getRGBValue(rl, gl, bl))
    pipe_higher_rgb = np.array(getRGBValue(rh, gh, bh))


    start =time.time()
    # build mask
    pipeEdge = cv2.inRange(imgData, pipe_lower_rgb, pipe_higher_rgb)
    pipeEdge[404:406,]=0

    # get value from bars
    bird_lower_rgb = np.array(getRGBValue(birdRl, birdGl, birdBl))
    bird_higher_rgb = np.array(getRGBValue(birdRh, birdGh, birdBh))
    # hard code of bird
    bird_lower_rgb = np.array(bird_lower_rgb)
    bird_higher_rgb = np.array(bird_higher_rgb)
    # build bird mask
    birdEdge = cv2.inRange(rgb, bird_lower_rgb, bird_higher_rgb)

    birdContours, hierarchy = cv2.findContours(image=birdEdge, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    pipeContours, hierarchy = cv2.findContours(image=pipeEdge, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    all_cnt_img = cv2.drawContours(np.copy(img), pipeContours, -1, (0,0 , 255), 2)
    all_cnt_img = cv2.drawContours(all_cnt_img, birdContours, -1, (255, 0,0 ), 2)
    pipeRect=[]
    final_result =  np.zeros((img.shape[0], img.shape[1], 3), np.float)
    if len(pipeContours) != 0:
        for cnt in pipeContours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            pipeRect.append([(x,y),(x + w, y + h)])
        pipeRect = fixRect(pipeRect)

    for each in pipeRect:
        cv2.rectangle(final_result, each[0], each[1], (0, 0, 255), thickness=1)

    if len(birdContours) != 0:
        for cnt in birdContours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(final_result, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

    cv2.rectangle(final_result, (0, 405), (288, 512), (0, 255, 255), thickness=1)

    print("time spend:",time.time()-start)
    cv2.imshow('Contours', all_cnt_img)
    cv2.imshow('mask', pipeEdge)
    cv2.imshow('experiment', img)
    cv2.imshow('final_result', final_result)
cv2.destroyAllWindows()

