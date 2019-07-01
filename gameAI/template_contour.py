import cv2
import numpy as np
import gameAI.config as config
import gameAI.discret_result as result


class TemplateContour:
    # A bird image to find in the image
    TEMPLATE_IMAGE = cv2.imread('redbird-downflap.png', 0)

    def __init__(self):
        # indicate the actual frame
        self._frame_count = 0

    def track_areas(self, frame):
        self.current_frame = frame
        self._frame_count = self._frame_count + 1
        # Track bird with template matching
        img_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        w, h = TemplateContour.TEMPLATE_IMAGE.shape[::-1]

        res = cv2.matchTemplate(img_gray, TemplateContour.TEMPLATE_IMAGE, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(TemplateContour.TEMPLATE_IMAGE, top_left, bottom_right, (255, 0, 0), cv2.FILLED)

        # Track for area with contours detection
        lower_hsv = np.array(config.get_contour_hsv_low())
        higher_hsv = np.array(config.get_contour_hsv_high())
        contour_hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2HSV)

        # build mask
        contours_mask = cv2.inRange(contour_hsv, lower_hsv, higher_hsv)

        # frame = cv2.bitwise_and(img, img, mask)
        contours, hierarchy = cv2.findContours(image=contours_mask, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        # all_cnt_img = cv2.drawContours(contours_frame, contours, -1, (255, 255, 0), 2)
        bird_area = (top_left, bottom_right)
        obstacle_area = []
        if len(contours) != 0:
            self._frame_count = self._frame_count + 1
            for cnt in contours:
                area = cv2.contourArea(cnt)
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 3)
                if not overlapping((top_left[0], top_left[1], w, h), (cx, cy, cw, ch)):
                    obstacle_area.append(((cx - 2, cy - 2), (cx + cw + 4, cy + ch + 4)))
        else:
            raise Exception("no result has been calculated")

        # obstacle_area = simplifyArea(obstacle_area)

        return result.DiscretizationResult(bird_area, simplifyArea(obstacle_area))


def simplifyArea(obstacles):
    newObstacle = []
    for ob1 in obstacles:
        if not isGround(ob1):
            for ob2 in obstacles:
                if not isGround(ob2) and not isEqual(ob1, ob2) and isBeside(ob1, ob2):
                    newObstacle.append(computeNewArea(ob1, ob2))
        else:
            newObstacle.append(ob1)
    return newObstacle


def computeNewArea(ob1, ob2):
    return (
        (
            min(ob1[0][0], ob2[0][0]), min(ob1[0][1], ob2[0][1])
        ),
        (
            max(ob1[1][0], ob2[1][0]), max(ob1[1][1], ob2[1][1])
        )
    )


def isGround(obstacle):
    return (obstacle[1][0] - obstacle[0][0]) * (obstacle[1][1] - obstacle[0][1]) > 30000


def isEqual(ob1, ob2):
    return ob1[0] == ob2[0] and ob1[1] == ob2[1]


def isBeside(ob1, ob2):
    wSpaceLeft = abs(ob1[0][0] - ob2[0][0])
    wSpaceRight = abs(ob1[1][0] - ob2[1][0])
    hSpace = min(abs(ob1[0][1] - ob2[1][1]),abs(ob2[1][1] - ob1[0][1]))
    return wSpaceLeft < 4 and wSpaceRight < 4 and hSpace <6



def overlapping(area1, area2):
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
