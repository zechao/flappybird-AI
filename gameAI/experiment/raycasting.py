import cv2
import numpy as np
from gameAI.experiment.Vector2d import Vector
from decimal import *

class Ray:
    def __init__(self, x, y, angle):
        self._pos = Vector(x, y)
        self._dir = Vector.fromAngle(angle)
    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, new_value):
        self._pos = new_value

    @property
    def dir(self):
        return self._dir

    @dir.setter
    def dir(self, new_value):
        self._dir = new_value

    def lookAt(self, x, y):
        self.dir.x = x - self.pos.x
        self.dir.y = y - self.pos.y

    def computeIntersect(self, wall):
        x1 = Decimal(wall.a.x)
        y1 = Decimal(wall.a.y)
        x2 = Decimal(wall.b.x)
        y2 = Decimal(wall.b.y)

        x3 = Decimal(self.pos.x)
        y3 = Decimal(self.pos.y)
        x4 = Decimal(self.pos.x) + Decimal(self.dir.x)
        y4 = Decimal(self.pos.y) + Decimal(self.dir.y)

        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den == 0:
            return Vector(0, 0)

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / den

        if t >= 0 and t <= 1 and u >= 0:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return Vector(x, y)
        else:
            return Vector(0, 0)

    def cast(self, walls):
        if isinstance(walls, Boundary):
            return self.computeIntersect(walls)
        elif isinstance(walls, tuple) or isinstance(walls, list):
            minDisPos = Vector(0, 0)
            minDis = float("inf")
            for wall in walls:
                vec = self.computeIntersect(wall)
                dis = Vector.distance(self.pos, vec)
                if dis <= minDis:
                    minDis = dis
                    minDisPos = vec
            return minDisPos
        else:
            NotImplemented

    def draw(self, image, length=0, color=[255] * 3):
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


class Sensor():
    def __init__(self, x, y, angles):
        self.pos = Vector(x, y)
        if isinstance(angles, int) or isinstance(angles, float):
            self.angles = [angles]
        elif isinstance(angles, tuple) or isinstance(angles, list):
            self.angles = list(angles)
        else:
            NotImplemented

        self.rays = [];
        for angle in self.angles:
            self.rays.append(Ray(x, y, np.radians(angle)))

    def draw(self, image, walls, color=[255] * 3):
        for ray in self.rays:
            pt = ray.cast(walls)
            if len(pt):
                cv2.line(image, self.pos.toIntTuple(), pt.toIntTuple(), [255, 0, 0])
                cv2.circle(image, pt.toIntTuple(), 5, [255] * 3, cv2.FILLED)


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


image = create_blank(1280, 720)
# Test the ray
# pos = []
# def onMouse(event, x, y, flags, param):
#     global pos
#     if event == cv2.EVENT_MOUSEMOVE:
#         pos = [x, y]

# cv2.namedWindow("raycasting", )
# cv2.setMouseCallback("raycasting", onMouse)
# r = Ray(200, 200,np.radians(-30))
# while True:
#     cv2.waitKey(1)
#     wall = Boundary(300, 100, 300, 300)
#     wall.draw(image)
#     if len(pos) > 0:
#         r.lookAt(pos[0], pos[1])
#     intersect = r.cast(wall)
#
#     cv2.circle(image, intersect.toIntTuple(), 5, [255] * 3, cv2.FILLED)
#     r.draw(image)
#     cv2.imshow("raycasting", image)


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
    walls = [Boundary(300, 0, 300, 300),
             Boundary(300, 300, 400, 300),
             Boundary(400, 300, 400, 0),

             Boundary(300, 400, 300, 720),
             Boundary(300, 400, 400, 400),
             Boundary(400, 400, 400, 720),

             Boundary(500, 100, 500, 600),
             Boundary(20, 100, 500, 600),
             ]


    sensor = Sensor(center.x, center.y, [-90,0])
    for wall in walls:
        wall.draw(image)
    sensor.draw(image, walls)
    cv2.imshow("raycasting", image)
