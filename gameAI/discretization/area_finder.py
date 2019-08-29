import cv2
import numpy as np
import gameAI.discretization.discret_result as result
import gameAI.discretization.geom2D as g2d
import time

# build color range for mask
PIPE_LOWER_RGB = np.array([84, 56, 71])
PIPE_HIGHER_RGB = np.array([85, 58, 73])

BIRD_LOWER_RGB = np.array([83, 56, 70])
BIRD_HIGHER_RGB = np.array([83, 56, 70])


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


class AreaFinder:
    def __init__(self):
        pass

    def track_areas(self, img):
        ret = result.DiscretizationResult(self.__findBirdArea(img), self.__findObstacles(img))
        return ret

    def __findObstacles(self,img):

        # mask areas
        pipeEdge = cv2.inRange(img, PIPE_LOWER_RGB, PIPE_HIGHER_RGB)
        pipeEdge[404:406, ] = 0

        pipeContours, hierarchy = cv2.findContours(image=pipeEdge, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)
        obstacle_area = []
        if len(pipeContours):
            rects = []
            for cnt in pipeContours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                rects.append([(x,y),(x+w,y+h)])
            rects =fixRect(rects)
            for start,end in rects:
                obstacle_area.append(g2d.Rect.fromPoints( start[0], start[1], end[0], end[1]))

        obstacle_area.append(g2d.Rect.fromPoints(0, 405, 288, 512))

        return obstacle_area

    def __findBirdArea(self,img):

        birdEdge = cv2.inRange(img, BIRD_LOWER_RGB, BIRD_HIGHER_RGB)

        birdContours, hierarchy = cv2.findContours(image=birdEdge, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_SIMPLE)

        if len(birdContours) != 0:
            cnt =birdContours[0]
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            return g2d.Rect.fromBoundingRect(x,y,w,h)
        else:
            RuntimeError("only one bird")