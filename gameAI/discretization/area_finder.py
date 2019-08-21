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
        if len(pipeContours) != 0:
            for cnt in pipeContours:
                area = cv2.contourArea(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                obstacle_area.append(g2d.Rect.fromBoundingRect( x, y, w, h))
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