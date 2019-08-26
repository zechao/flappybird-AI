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

def fixRect(pipeRect):
    length = len(pipeRect)
    res=[]
    discard=[]
    for i in range(length):
        findDisconnection=False
        for j in range (i+1,length):
            if i not in discard and j not in discard:
                h1 = abs(pipeRect[i][0][1] - pipeRect[i][1][1])
                h2 = abs(pipeRect[j][0][1] - pipeRect[j][1][1])
                w1 = abs(pipeRect[i][0][0] - pipeRect[i][1][0])
                w2 = abs(pipeRect[i][0][0] - pipeRect[i][1][0])
                if h1 == h2 and w1 == w2 and w1 == 2:
                    findDisconnection = True
                    x1 = min(pipeRect[i][0][0], pipeRect[j][0][0])
                    y1 = min(pipeRect[i][0][1], pipeRect[j][0][1])
                    x2 = max(pipeRect[i][1][0], pipeRect[j][1][0])
                    y2 = max(pipeRect[i][1][1], pipeRect[j][1][1])
                    res.append([(x1, y1), (x2, y2)])
                    discard.append(i)
                    discard.append(j)
        if not findDisconnection and i not in discard and j not in discard:
            res.append(pipeRect[i])
    points =np.array([[[126,0]], [[128,51]],[[172, 0]],[[174, 51]],[[176,160]]],np.float32)
    x, y, w, h = cv2.boundingRect(points)
    res.append([(x,y),(x + w, y + h)])
    return res


createRGBTrackerBar()

imgName = 'frame1000.png'
img = cv2.imread(imgName)
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
setDefaultHSVValue()

while True:
    cv2.waitKey(int(1000 / 60))
    img = cv2.imread('frame1000.png')
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

