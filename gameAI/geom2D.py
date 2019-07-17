from random import *
from math import *


class Vector:
    def __init__(self, x=0, y=0):
        self.x = 0
        self.y = 0
        if isinstance(x, tuple) or isinstance(x, list):
            y = x[1]
            x = x[0]
        elif isinstance(x, Vector):
            y = x.y
            x = x.x

        self.set(x, y)

    @staticmethod
    def random(size=1):
        sizex = size
        sizey = size
        if isinstance(size, tuple) or isinstance(size, list):
            sizex = size[0]
            sizey = size[1]
        elif isinstance(size, Vector):
            sizex = size.x
            sizey = size.y
        return Vector(random() * sizex, random() * sizey)

    @staticmethod
    def randomUnitCircle():
        d = random() * pi
        return Vector(cos(d) * choice([1, -1]), sin(d) * choice([1, -1]))

    @staticmethod
    def distance(a, b):
        return (a - b).getLength()

    @staticmethod
    def angle(v1, v2):
        return acos(v1.dot(v2) / (v1.getLength() * v2.getLength()))

    @staticmethod
    def fromAngle(a):
        return Vector(cos(a), sin(a))

    @staticmethod
    def angleDeg(v1, v2):
        return Vector.angle(v1, v2) * 180.0 / pi

    def set(self, x, y):
        self.x = x
        self.y = y

    def getRotated(self, a):
        x = self.x * cos(a) - self.y * sin(a)
        y = self.x * sin(a) + self.y * cos(a)
        return Vector(x, y)

    def toArr(self):
        return [self.x, self.y]

    def toTuple(self):
        return (self.x, self.y)

    def toIntTuple(self):
        return self.toInt().toTuple()

    def toInt(self):
        return Vector(int(self.x), int(self.y))

    def toIntArr(self):
        return self.toInt().toArr()

    def normalized(self):
        norm = self.getNormalized()
        self.x = norm.x
        self.y = norm.y

    def getNormalized(self):
        length = self.getLength()
        if length != 0:
            return Vector(self.x / length, self.y / length)
        else:
            return NotImplemented

    def dot(self, other):
        if isinstance(other, Vector):
            return self.x * other.x + self.y * other.y
        elif isinstance(other, tuple) or isinstance(other, list):
            return self.x * other[0] + self.y * other[1]
        else:
            return NotImplemented

    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x + other.x, self.y + other.y)
        elif isinstance(other, tuple) or isinstance(other, list):
            return Vector(self.x + other[0], self.y + other[1])
        elif isinstance(other, int) or isinstance(other, float):
            return Vector(self.x + other, self.y + other)
        else:
            return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x - other.x, self.y - other.y)
        if isinstance(other, tuple) or isinstance(other, list):
            return Vector(self.x - other[0], self.y - other[1])
        elif isinstance(other, int) or isinstance(other, float):
            return Vector(self.x - other, self.y - other)
        else:
            return NotImplemented

    def __rsub__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x - self.x, other.y - self.y)
        elif isinstance(other, tuple) or isinstance(other, list):
            return Vector(other[0] - self.x, other[1] - self.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector(other - self.x, other - self.y)
        else:
            return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.x * other.x, self.y * other.y)
        elif isinstance(other, tuple) or isinstance(other, list):
            return Vector(self.x * other[0], self.y * other[1])
        elif isinstance(other, int) or isinstance(other, float):
            return Vector(self.x * other, self.y * other)
        else:
            return NotImplemented

    def __div__(self, other):
        print("hola")
        if isinstance(other, Vector):
            return Vector(self.x / other.x, self.y / other.y)
        elif isinstance(other, tuple) or isinstance(other, list):
            return Vector(self.x / other[0], self.y / other[1])
        elif isinstance(other, int) or isinstance(other, float):
            return Vector(self.x / other, self.y / other)
        else:
            return NotImplemented

    def __rdiv__(self, other):
        if isinstance(other, Vector):
            return Vector(other.x / self.x, other.y / self.y)
        elif isinstance(other, tuple) or isinstance(other, list):
            return Vector(other[0] / self.x, other[1] / self.y)
        elif isinstance(other, int) or isinstance(other, float):
            return Vector(other / self.x, other / self.y)
        else:
            return NotImplemented

    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            return Vector(self.x ** other, self.y ** other)
        else:
            return NotImplemented

    def __iadd__(self, other):
        if isinstance(other, Vector):
            self.x += other.x
            self.y += other.y
            return self
        elif isinstance(other, tuple) or isinstance(other, list):
            self.x += other[0]
            self.y += other[1]
            return self
        elif isinstance(other, int) or isinstance(other, float):
            self.x += other
            self.y += other
            return self
        else:
            return NotImplemented

    def __isub__(self, other):
        if isinstance(other, Vector):
            self.x -= other.x
            self.y -= other.y
            return self
        elif isinstance(other, tuple) or isinstance(other, list):
            self.x -= other[0]
            self.y -= other[1]
            return self
        elif isinstance(other, int) or isinstance(other, float):
            self.x -= other
            self.y -= other
            return self
        else:
            return NotImplemented

    def __imul__(self, other):
        if isinstance(other, Vector):
            self.x *= other.x
            self.y *= other.y
            return self
        elif isinstance(other, tuple) or isinstance(other, list):
            self.x *= other[0]
            self.y *= other[1]
            return self
        elif isinstance(other, int) or isinstance(other, float):
            self.x *= other
            self.y *= other
            return self
        else:
            return NotImplemented

    def __idiv__(self, other):
        if isinstance(other, Vector):
            self.x /= other.x
            self.y /= other.y
            return self
        elif isinstance(other, tuple) or isinstance(other, list):
            self.x /= other[0]
            self.y /= other[1]
            return self
        elif isinstance(other, int) or isinstance(other, float):
            self.x /= other
            self.y /= other
            return self
        else:
            return NotImplemented

    def __ipow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            self.x **= other
            self.y **= other
            return self
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        else:
            return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Vector):
            return self.x != other.x or self.y != other.y
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Vector):
            return self.getLength() > other.getLength()
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Vector):
            return self.getLength() >= other.getLength()
        else:
            return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Vector):
            return self.getLength() < other.getLength()
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, Vector):
            return self.getLength() <= other.getLength()
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Vector):
            return self.x == other.x and self.y == other.y
        else:
            return NotImplemented

    def __len__(self):
        return int(sqrt(self.x ** 2 + self.y ** 2))

    def getLength(self):
        return sqrt(self.x ** 2 + self.y ** 2)

    def __getitem__(self, key):
        if key == "x" or key == "X" or key == 0 or key == "0":
            return self.x
        elif key == "y" or key == "Y" or key == 1 or key == "1":
            return self.y

    def __str__(self):
        return "[x: %(x)f, y: %(y)f]" % self

    def __repr__(self):
        return "{'x': %(x)f, 'y': %(y)f}" % self

    def __neg__(self):
        return Vector(-self.x, -self.y)


class Rect():
    def __init__(self, leftTop, rightDown):
        self._leftTop = leftTop

        self._rightDown = rightDown

        self._rightTop = Vector(rightDown.x, leftTop.y)

        self._leftDown = Vector(leftTop.x, rightDown.y)

        self._topCenter = Vector(leftTop.x + (rightDown.x - leftTop.x) / 2, leftTop.y)
        self._downCenter = Vector(leftTop.x + (rightDown.x - leftTop.x) / 2, rightDown.y)
        self._leftCenter = Vector(leftTop.x, leftTop.y + (rightDown.y - leftTop.y) / 2)
        self._rightCenter = Vector(rightDown.x, leftTop.y + (rightDown.y - leftTop.y) / 2)
        self._center = Vector(leftTop.x + (rightDown.x - leftTop.x) / 2, leftTop.y + (rightDown.y - leftTop.y) / 2)
        self._width = rightDown.y - leftTop.y
        self._height = rightDown.x - leftTop.x

    @staticmethod
    def fromPoints(x1, y1, x2, y2):
        return Rect(Vector(x1, y1), Vector(x2, y2))

    @staticmethod
    def fromBoundingRect(x1, y1, width, height):
        return Rect(Vector(x1, y1), Vector(x1 + width, y1 + height))

    @property
    def leftTop(self):
        return self._leftTop

    @property
    def rightTop(self):
        return self._rightTop

    @property
    def rightDown(self):
        return self._rightDown

    @property
    def leftDown(self):
        return self._leftDown

    @property
    def topCenter(self):
        return self._topCenter

    @property
    def rightCenter(self):
        return self._rightCenter

    @property
    def downCenter(self):
        return self._downCenter

    @property
    def leftCenter(self):
        return self._leftCenter

    @property
    def center(self):
        return self._center

    @property
    def p1(self):
        return self.leftTop.toIntTuple()

    @property
    def p2(self):
        return self.rightDown.toIntTuple()

    def overlapping(self, other):
        """
        Check if given two areas are overlapping
        :param other: Rect
        :return: bool
        """
        return not (self.rightTop.x < other.leftTop.x or
                    self.rightDown.y < other.leftTop.y or
                    self.leftTop.x > other.rightTop.x or
                    self.leftTop.y > other.rightDown.y)

    def __eq__(self, other):
        if isinstance(other, Rect):
            return self.leftTop == other.leftTop and self.rightDown == self.rightDown
        else:
            return False
