import cv2
import numpy as np
from gameAI.discretization.geom2D import Vector


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

    @staticmethod
    def fromVector(v1, v2, boundType=BoundType.BLANK):
        return GameBoundary(v1.x, v1.y, v2.x, v2.y, boundType)

    def draw(self, img, color=[255] * 3):
        if not None:
            if self._boundType == BoundType.BORDER:
                cv2.line(img, self.a.toIntTuple(), self.b.toIntTuple(), [255, 255, 255], 2)
            elif self._boundType == BoundType.OBSTACLE:
                cv2.line(img, self.a.toIntTuple(), self.b.toIntTuple(), [0, 0, 255], 2)
            elif self._boundType == BoundType.BLANK:
                cv2.line(img, self.a.toIntTuple(), self.b.toIntTuple(), [255, 0, 0], 2)


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

    @pos.setter
    def pos(self, value):
        self._rayPos = value

    def draw(self, img):
        if img is not None:
            if self._boundType == BoundType.BORDER:
                cv2.line(img, self._rayPos.toIntTuple(), self._hitPoint.toIntTuple(), [255, 255, 0], 1)
                cv2.circle(img, self._hitPoint.toIntTuple(), 3, [255, 255, 0], 2)
                # cv2.putText(img, str(res.distance), res.hitPoint.toIntTuple(), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 0])
            elif self._boundType == BoundType.OBSTACLE:
                cv2.line(img, self._rayPos.toIntTuple(), self._hitPoint.toIntTuple(), [255, 0, 0], 1)
                cv2.circle(img, self._hitPoint.toIntTuple(), 3, [0, 0, 255], 2)
                # cv2.putText(img, str(res.distance), res.hitPoint.toIntTuple(), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255])
            elif self._boundType == BoundType.BLANK:
                cv2.line(img, self._rayPos.toIntTuple(), self._hitPoint.toIntTuple(), [255, 255, 255], 1)
                # cv2.circle(img, res.toIntTuple(), 3, [255, 255, 255], 2)


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

    @staticmethod
    def createBlankSensor(angle):
        return Sensor(0, 0, angle)

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
            return SensorResult(self.pos, self._ray.angle, hitPoint, Vector.distance(self.pos, hitPoint),
                                wall.boundType)

    def castWalls(self, walls):
        """
        cast each walls
        :param walls:
        :return: the sensor result between a wall or list of walls, in case of return the nearest intersect point
        """
        if isinstance(walls, GameBoundary):
            return self.__castWall(walls)
        elif isinstance(walls, tuple) or isinstance(walls, list):
            minDisHitSensorR = None
            minDis = float("inf")
            for wall in walls:
                if isinstance(wall, GameBoundary):
                    sensorR = self.__castWall(wall)
                    if sensorR == None:
                        continue
                    if sensorR.distance <= minDis:
                        minDis = sensorR.distance
                        minDisHitSensorR = sensorR
                else:
                    return None
            if minDisHitSensorR is None:
                self.lastMinDisHitSensorR.pos = self.pos
                return self.lastMinDisHitSensorR
            else:
                self.lastMinDisHitSensorR = minDisHitSensorR
                return minDisHitSensorR
        else:
            NotImplemented

    def castAndDraw(self, walls, img=None):
        res = self.castWalls(walls)
        if res is None:
            return res
        if img is None:
            return res
        self.draw(res, img)
        return res

    def customPositionCast(self, x, y, walls, img=None):
        self._pos = Vector(x, y)
        self._ray = Ray(x, y, self.angle)
        self.res = self.castWalls(walls)
        if img is not None:
            self.draw(self.res, img)
        return self.res

    def draw(self, res, img):
        if img is not None and res is not None:
            if res.boundType == BoundType.BORDER:
                cv2.line(img, res.pos.toIntTuple(), res.hitPoint.toIntTuple(), [255, 255, 0], 1)
                cv2.circle(img, res.hitPoint.toIntTuple(), 3, [255, 255, 0], 2)
                # cv2.putText(img, str(res.distance), res.hitPoint.toIntTuple(), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 0])
            elif res.boundType == BoundType.OBSTACLE:
                cv2.line(img, res.pos.toIntTuple(), res.hitPoint.toIntTuple(), [255, 0, 0], 1)
                cv2.circle(img, res.hitPoint.toIntTuple(), 3, [0, 0, 255], 2)
                # cv2.putText(img, str(res.distance), res.hitPoint.toIntTuple(), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255])
            elif res.boundType == BoundType.BLANK:
                cv2.line(img, res.pos.toIntTuple(), res.hitPoint.toIntTuple(), [255, 255, 255], 1)
                # cv2.circle(img, res.toIntTuple(), 3, [255, 255, 255], 2)


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
        cv2.waitKey(20)
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
        sensor5 = Sensor(center.x, center.y, 45)
        sensor6 = Sensor.createBlankSensor(-135)
        sensor7 = Sensor.createBlankSensor(135)
        sensor4 = Sensor(center.x, center.y, -45)
        for wall in walls:
            wall.draw(image)
        sensor6.customPositionCast(center.x, center.y, walls, image)
        sensor2.castAndDraw(walls, image)
        sensor1.castAndDraw(walls, image)
        sensor3.castAndDraw(walls, image)
        sensor4.castAndDraw(walls, image)
        sensor5.castAndDraw(walls, image)
        res = sensor7.customPositionCast(center.x, center.y, walls)
        res.draw(image)
        cv2.imshow("raycasting", image)
