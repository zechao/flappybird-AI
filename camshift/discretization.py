import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy





def draw_bird_backProject(wname, dst):
    cv2.imshow(wname, dst)


def get_bound_box(ret):
    box = cv2.boxPoints(ret)
    box = np.int0(box)
    return box


def get_bird_hsv_low():
    return np.array((0., 237., 204.))


def get_bird_hsv_high():
    return np.array((170., 255., 255.))


def get_contour_hsv_low():
    return np.array((0., 0., 95.))


def get_contour_hsv_high():
    return np.array((170., 255., 255.))


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


class CamShiftTracking:
    # hard code of the area which start the tracking
    __bird_body_area = np.array((70, 5, 248, 10))

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    __term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


    def __init__(self, frame):
        c, w, r, h = self.__bird_body_area
        self.track_window = (c, r, w, h)

        # draw rectangle to the tracked area
        # cv2.rectangle(game_frame, (c, r), (c + w, r + h), (0, 255, 0), 1)

        self.roi = frame[r:r + h, c:c + w]
        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_RGB2HSV)

        # mask area for bird body color
        self.mask = cv2.inRange(self.hsv_roi, get_bird_hsv_low(), get_bird_hsv_high())
        # calc histogram for bird body mask
        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

    def track_area(self, frame):
        """

        :param frame: a current image frame
        :return: an area tracked by the Camshift algorithm
        """
        hsv = cv2.cvtColor(bird_frame, cv2.COLOR_RGB2HSV)

        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # apply CamShift to get the new location
        ret, self.track_window = cv2.CamShift(dst, self.track_window, self.__term_criteria)

        (x, y, w, h) = (int(ret[0][0]), int(ret[0][1]), int(ret[1][0]), int(ret[1][1]))
        # expanded bird tracked size, and draw area
        # cv2.rectangle(bird_frame, (x - 10, y - 10), (x + w + 5, y + h - 5), (0, 255, 0), cv2.FILLED)
        self.result_area = (x - 12, y - 12), (x + w + 7, y + h - 7)
        return self.result_area




# init game and get first frame
game = flappy.GameState()

game_frame = game.next_frame(False)

# select area of bird, hardcoded!

camshift = CamShiftTracking(game_frame)

run = False
flap = False

while True:
    flap = False
    k = cv2.waitKey(30) & 0xFF  # when using 64bit machine
    if k == 27:  # wait for ESC key to exit
        break
    if k == ord('1'):
        run = True
    if k == ord('2'):
        run = False
    if k == ord('w'):
        run = False
        game_frame = game.next_frame(True)
    if k == ord('s'):
        run = False
        game_frame = game.next_frame(False)
    if k == 32:
        flap = True
    if run:
        game_frame = game.next_frame(flap)

    # copy of game frame for bird tracking
    bird_frame = np.copy(game_frame)
    # copy for contours tracking
    contours_frame = np.copy(game_frame)
    # copy for template tracking
    template_frame = np.copy(game_frame)

    cv2.imshow("Flappy Bird", game_frame)
    ret = True
    if run:

        result_area= camshift.track_area(bird_frame)
        cv2.rectangle(bird_frame, result_area[0], result_area[1], (0, 255, 0), cv2.FILLED)
        cv2.imshow("Flappy Bird",bird_frame)

        # draw_bird_backProject("BackProject", dst)

        # Track bird with template matching
        img_gray = cv2.cvtColor(template_frame, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('redbird-downflap.png', 0)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(template_frame, top_left, bottom_right, (255, 0, 0), cv2.FILLED)

        # Track for area with contours detection
        lower_hsv = np.array(get_contour_hsv_low())
        higher_hsv = np.array(get_contour_hsv_high())
        contour_hsv = cv2.cvtColor(game_frame, cv2.COLOR_RGB2HSV)
        # build mask
        contours_mask = cv2.inRange(contour_hsv, lower_hsv, higher_hsv)
        # frame = cv2.bitwise_and(img, img, mask)
        contours, hierarchy = cv2.findContours(image=contours_mask, mode=cv2.RETR_EXTERNAL,
                                               method=cv2.CHAIN_APPROX_SIMPLE)
        all_cnt_img = cv2.drawContours(contours_frame, contours, -1, (255, 255, 0), 2)
        if len(contours) != 0:
            for cnt in contours:
                area = cv2.contourArea(cnt)
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                # cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 3)
                if isOverLapping((top_left[0], top_left[1], w, h), (cx, cy, cw, ch)):
                    cv2.rectangle(contours_frame, top_left, bottom_right, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.rectangle(contours_frame, (cx - 2, cy - 2), (cx + cw + 4, cy + ch + 4), (0, 0, 255), cv2.FILLED)
        cv2.imshow('contours', contours_frame)

game.quit()
cv2.destroyAllWindows()
