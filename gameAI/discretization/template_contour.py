import cv2
import numpy as np
import gameAI.discretization.config as config
import gameAI.discretization.discret_result as result
import gameAI.discretization.geom2D as g2d


class TemplateContour:
    # A bird image to find in the image
    TEMPLATE_IMAGE = cv2.imread('./template/redbird-downflap.png', 0)

    def __init__(self):
        # indicate the actual frame
        self._frame_count = 0

    def track_areas(self, frame):
        self.current_frame = np.copy(frame)
        self._frame_count = self._frame_count + 1
        # Track bird with template matching
        img_gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
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

        # calc the fixed area of our contours
        # all_cnt_img = cv2.drawContours(contours_frame, contours, -1, (255, 255, 0), 2)
        bird_area = g2d.Rect(g2d.Vector(top_left), g2d.Vector(bottom_right))
        obstacle_area = []
        if len(contours) != 0:
            self._frame_count = self._frame_count + 1
            for cnt in contours:
                area = cv2.contourArea(cnt)
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                if not bird_area.overlapping(g2d.Rect.fromBoundingRect(cx, cy, cw, ch)):
                    # add extra pixel to ensure
                    obstacle_area.append(g2d.Rect.fromBoundingRect(cx - 2, cy - 2, cw + 4, ch + 4))
        else:
            raise Exception("no result has been calculated")

        return result.DiscretizationResult(bird_area, simplifyArea(obstacle_area))


def simplifyArea(obstacles):
    newObstacle = []
    for rec1 in obstacles:
        if not isGround(rec1):
            for rec2 in obstacles:
                if not isGround(rec2) and not rec1 == rec2 and isBeside(rec1, rec2):
                    newObstacle.append(computeNewArea(rec1, rec2))
        else:
            newObstacle.append(rec1)
    return newObstacle


def computeNewArea(rect1, rect2):
    return g2d.Rect.fromPoints(
        min(rect1.leftTop.x, rect2.leftTop.x),
        min(rect1.leftTop.y, rect2.leftTop.y),
        max(rect1.rightDown.x, rect2.rightDown.x),
        max(rect1.rightDown.y, rect2.rightDown.y)
    )


def isGround(obstacle):
    return (obstacle.rightDown.x - obstacle.leftTop.x) * (obstacle.rightDown.y - obstacle.leftTop.y) > 30000


def isBeside(rec1, rec2):
    wSpaceLeft = abs(rec1.leftTop.x - rec2.leftTop.x)
    wSpaceRight = abs(rec1.rightDown.x - rec2.rightDown.x)
    hSpace = min(abs(rec1.leftTop.y - rec2.rightDown.y), abs(rec2.rightDown.y - rec1.leftTop.y))
    return wSpaceLeft < 4 and wSpaceRight < 4 and hSpace < 6
