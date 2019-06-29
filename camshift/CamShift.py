import numpy as np
import cv2
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
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)
        # apply CamShift to get the new location
        ret, self.track_window = cv2.CamShift(dst, self.track_window, self.__term_criteria)

        (x, y, w, h) = (int(ret[0][0]), int(ret[0][1]), int(ret[1][0]), int(ret[1][1]))
        # expanded bird tracked size, and draw area
        # cv2.rectangle(bird_frame, (x - 10, y - 10), (x + w + 5, y + h - 5), (0, 255, 0), cv2.FILLED)
        self.result_area = (x - 12, y - 12), (x + w + 7, y + h - 7)
        return self.result_area
