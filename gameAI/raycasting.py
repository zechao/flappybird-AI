import cv2
import numpy as np
from gameAI.experiment.Vector2d import Vector
from decimal import *


class Ray:
    def __init__(self, x, y, angle):
        self._pos = Vector(x, y)
        self._dir = Vector.fromAngle(np.radians(angle)).getNormalized()
        self._angle = angle

    @property
    def pos(self):
        return self._pos

    @property
    def dir(self):
        return self._dir

    @property
    def angle(self):
        return self._angle

    def lookAt(self, angle):
        self._dir = Vector.fromAngle(np.radians(angle)).getNormalized()
        self._angle = angle

    def computeHitPoint(self, wall):
        x1 = wall.a.x
        y1 = wall.a.y
        x2 = wall.b.x
        y2 = wall.b.y

        x3 = self.pos.x
        y3 = self.pos.y
        x4 = self.pos.x + self.dir.x
        y4 = self.pos.y + self.dir.y

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if t >= 0 and t <= 1 and u >= 0:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return Vector(x, y)
        else:
            return None

    def draw(self, image, length=50, color=[255] * 3):
        return cv2.line(image, self.pos.toIntTuple(), (self.pos + self.dir * length).toIntTuple(), color)


class Boundary:
    def __init__(self, x1, y1, x2, y2):
        self._a = Vector(x1, y1)
        self._b = Vector(x2, y2)

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, new_value):
        self._a = new_value

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, new_value):
        self._b = new_value

    def draw(self, image, color=[255] * 3):
        return cv2.line(image, self.a.toTuple(), self.b.toTuple(), color)


from enum import Enum


class BoundType(Enum):
    OBSTACLE = 1
    BORDER = 2
    BLANK = 3


class GameBoundary(Boundary):
    def __init__(self, x1, y1, x2, y2, boundType=BoundType.BLANK):
        super(GameBoundary, self).__init__(x1, y1, x2, y2)
        self._boundType = boundType

    @property
    def boundType(self):
        return self._boundType


class SensorResult():
    """
    Result of Sensor, and also the input of our neural network
    """

    def __init__(self, rayPos, rayAngle, hitPoint, distance, boundType):
        self._hitPoint = hitPoint
        self._rayPos = rayPos
        self._rayAngle = rayAngle
        self._distance = distance
        self._boundType = boundType

    @property
    def pos(self):
        return self._rayPos

    @property
    def hitPoint(self):
        return self._hitPoint

    @property
    def distance(self):
        return self._distance

    @property
    def boundType(self):
        return self._boundType

    @property
    def angle(self):
        return self._angle


class Sensor():
    def __init__(self, x, y, angle):
        """
        :param x: sensor pos.x
        :param y: sensor pos.y
        :param angle: an angle  between 0-360, it's direction of each ray
        """
        self._pos = Vector(x, y)
        self._angle = angle
        self._ray = Ray(x, y, angle)

    @property
    def pos(self):
        return self._pos

    @property
    def angle(self):
        return self._angle

    def __castWall(self, wall):
        """
        cast one wall
        :param wall:
        :return: the sensor result between one wall
        """
        hitPoint = self._ray.computeHitPoint(wall)
        if hitPoint == None:
            return None
        else:
            return SensorResult(self.pos, self._ray.angle, hitPoint, Vector.distance(self.pos,hitPoint), wall.boundType)

    def castWalls(self, walls):
        """
        cast each walls
        :param walls:
        :return: the sensor result between a wall or list of walls, in case of return the nearest intersect point
        """
        if isinstance(walls, GameBoundary):
            return self.__castWall(walls)
        elif isinstance(walls, tuple) or isinstance(walls, list):
            minDisHitSensor = None
            minDis = float("inf")
            for wall in walls:
                if isinstance(wall, GameBoundary):
                    sensorR = self.__castWall(wall)
                    if sensorR == None:
                        continue
                    if sensorR.distance <= minDis:
                        minDis = sensorR.distance
                        minDisHitSensor = sensorR
                else:
                    return None
            return minDisHitSensor
        else:
            NotImplemented

    def draw(self, img, walls):
        res = self.castWalls(walls)
        if not None:
            if res.boundType == BoundType.BORDER:
                cv2.line(img, res.pos.toIntTuple(), res.hitPoint.toIntTuple(), [255, 255, 0], 1)
                cv2.circle(img, res.hitPoint.toIntTuple(), 3, [255, 255, 0], 2)
            elif res.boundType == BoundType.OBSTACLE:
                cv2.line(img, res.pos.toIntTuple(), res.hitPoint.toIntTuple(), [255, 0, 0], 1)
                cv2.circle(img, res.hitPoint.toIntTuple(), 3, [255, 0, 0], 2)
            elif res.boundType == BoundType.BLANK:
                cv2.line(img, res.pos.toIntTuple(), res.hitPoint.toIntTuple(), [255, 255, 255], 1)
                cv2.circle(img, res.toIntTuple(), 3, [255, 255, 255], 2)


class SensorGroup():
    def __init__(self, x, y, angles):
        """
        :param x: sensor pos.x
        :param y: sensor pos.y
        :param angles: an angle or list of angles between 0-360, it's direction of each ray
        """
        self.pos = Vector(x, y)
        if isinstance(angles, int) or isinstance(angles, float):
            self.angles = [angles]
        elif isinstance(angles, tuple) or isinstance(angles, list):
            self.angles = list(angles)
        else:
            NotImplemented

        self.rays = [];
        for angle in self.angles:
            self.rays.append(Sensor(x, y, angle))

    def castAllRay(self, walls):
        res = {}
        for ray in self.rays:
            r = self.rayCast(ray, walls)
            res.update(ray.angle)
        return res


def detectWall(self):
    for ray in self.rays:
        self.intersectPoint, self.distance = ray.cast(walls)


def draw(self, image, walls, color=[255] * 3):
    if len(self.intersectPoint):
        cv2.line(image, self.pos.toIntTuple(), self.intersectPoint.toIntTuple(), color)
        cv2.circle(image, self.intersectPoint.toIntTuple(), 5, color, cv2.FILLED)


def create_blank(width, height, rgb_color=(0, 0, 0)):
    """Create new image(numpy array) filled with certain color in RGB"""
    # Create black blank image
    image = np.zeros((height, width, 3), np.float)

    # Since OpenCV uses BGR, convert the color first
    color = tuple(reversed(rgb_color))
    # Fill image with color
    image[:] = color
    return image


def createAngleList(aStart=0, aEnd=360, rayNum=72):
    return list(range(aStart, aEnd, int((aEnd - aStart) / (rayNum - 1)))) + [aEnd]


if __name__ == '__main__':
    pos = []

    center = Vector(200, 200)


    def onMouse(event, x, y, flags, param):
        global pos
        global center
        if event == cv2.EVENT_LBUTTONDBLCLK:
            pos.append(Vector(x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            center = Vector(x, y)


    cv2.namedWindow("raycasting", )
    cv2.setMouseCallback("raycasting", onMouse)

    while True:
        cv2.waitKey(30)
        image = create_blank(1280, 720)
        walls = [GameBoundary(300, 0, 300, 300, BoundType.OBSTACLE),
                 GameBoundary(300, 300, 400, 300, BoundType.OBSTACLE),
                 GameBoundary(400, 300, 400, 0, BoundType.OBSTACLE),

                 GameBoundary(300, 400, 300, 720, BoundType.OBSTACLE),
                 GameBoundary(300, 400, 400, 400, BoundType.OBSTACLE),
                 GameBoundary(400, 400, 400, 720, BoundType.OBSTACLE),

                 GameBoundary(500, 100, 500, 600, BoundType.OBSTACLE),

                 GameBoundary(0, 0, 1280, 0, BoundType.BORDER),
                 GameBoundary(1280, 0, 1280, 720, BoundType.BORDER),
                 GameBoundary(1280, 720, 0, 720, BoundType.BORDER),
                 GameBoundary(0, 720, 0, 0, BoundType.BORDER),
                 ]

        sensor1 = Sensor(center.x, center.y, 90)
        sensor2 = Sensor(center.x, center.y, -90)
        sensor3 = Sensor(center.x, center.y, 0)
        sensor4 = Sensor(center.x, center.y, 180)
        for wall in walls:
            wall.draw(image)
        sensor1.draw(image, walls)

        sensor2.draw(image, walls)

        sensor3.draw(image, walls)

        sensor4.draw(image, walls)
        cv2.imshow("raycasting", image)
