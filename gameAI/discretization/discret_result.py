"""
This class return the _result of the area found after image discretization, which contain an area of
bird and obstacles area list
"""
import cv2
import gameAI.discretization.sensor as sr


class DiscretizationResult:
    def __init__(self, birdArea=None, obstaclesArea=None, ):
        if obstaclesArea is None:
            obstaclesArea = []
        self._birdArea = birdArea
        self._obstaclesArea = obstaclesArea

    @property
    def birdArea(self):
        return self._birdArea

    @property
    def obstacleArea(self):
        return self.obstacles_area

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = value

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = value

    def getAreaImage(self, image):
        cv2.rectangle(image, self._birdArea.p1, self._birdArea.p2, (255, 0, 0), cv2.FILLED)
        for obstacle in self._obstaclesArea:
            cv2.rectangle(image, obstacle.p1, obstacle.p2, (0, 0, 255), cv2.FILLED)
        return image

    def getGameWalls(self, width, height):
        self.bounds = []
        if len(self.bounds) == 0:
            for obstacle in self._obstaclesArea:
                self.bounds.append(
                    sr.GameBoundary.fromVector(obstacle.leftTop, obstacle.rightTop, sr.BoundType.OBSTACLE))
                self.bounds.append(
                    sr.GameBoundary.fromVector(obstacle.rightTop, obstacle.rightDown, sr.BoundType.OBSTACLE))
                self.bounds.append(
                    sr.GameBoundary.fromVector(obstacle.rightDown, obstacle.leftDown, sr.BoundType.OBSTACLE))
                self.bounds.append(
                    sr.GameBoundary.fromVector(obstacle.leftDown, obstacle.leftTop, sr.BoundType.OBSTACLE))

            self.bounds.append(sr.GameBoundary(0, 0, width, 0, sr.BoundType.BORDER))
            self.bounds.append(sr.GameBoundary(height, 0, height, width, sr.BoundType.BORDER))
        return self.bounds

    def getBirdFrontCenter(self):
        return self._birdArea.rightCenter


