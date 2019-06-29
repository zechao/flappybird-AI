"""
This class return the result of the area found after image discretization, which contain an area of
bird and obstacles area list
"""
import numpy as np


class DiscretizationResult:

    def __init__(self, bird_area=None, *obstacles_area):
        self._bird_area = bird_area
        self._obstacles_area = obstacles_area

    @property
    def bird_area(self):
        return self._bird_area

    @property
    def obstacle_area(self):
        return self.obstacles_area

    def is_collided(self):
        for obstacle in self._obstacles_area:
            if isOverLapping(self._bird_area, obstacle):
                return True
        return False


def isOverLapping(area1, area2):
    """
    Check if given two areas are overlapping
    :param area1: np.array
    :param area2: np.array
    :return: bool
    """
    r1x = area1[0]
    r1y = area1[1]
    r1width = area1[2]
    r1height = area1[3]

    r2x = area2[0]
    r2y = area2[1]
    r2width = area2[2]
    r2height = area2[3]

    return not (r1x + r1width < r2x or
                r1y + r1height < r2y or
                r1x > r2x + r2width or
                r1y > r2y + r2height)
