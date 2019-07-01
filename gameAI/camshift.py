import numpy as np
import cv2
import gameAI.config as config
import gameAI.discret_result as result


class CamShiftTracking:
    # hard code of the area which start the tracking
    BIRD_BODY_AREA = np.array((70, 5, 248, 10))

    # Setup the termination criteria, either 10 iteration or move by at least 1 pt
    TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    def __init__(self):
        # indicate the actual frame
        self._frame_count = 0

    def __init_track(self):
        c, w, r, h = CamShiftTracking.BIRD_BODY_AREA
        self.track_window = (c, r, w, h)

        self.roi = self.current_frame[r:r + h, c:c + w]
        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_RGB2HSV)

        # mask area for bird body color
        self.mask = cv2.inRange(self.hsv_roi, config.get_bird_hsv_low(), config.get_bird_hsv_high())
        # calc histogram for bird body mask

        self.roi_hist = cv2.calcHist([self.hsv_roi], [0], self.mask, [180], [0, 180])
        cv2.normalize(self.roi_hist, self.roi_hist, 0, 255, cv2.NORM_MINMAX)

        return self.__track_area()

    def __track_area(self):
        hsv = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2HSV)

        dst = cv2.calcBackProject([hsv], [0], self.roi_hist, [0, 180], 1)

        # apply CamShift to get the new location

        ret, self.track_window = cv2.CamShift(dst, self.track_window, CamShiftTracking.TERM_CRITERIA)
        (x, y, w, h) = (int(ret[0][0]), int(ret[0][1]), int(ret[1][0]), int(ret[1][1]))
        while all(v == 0 for v in (x, y, w, h)):
            ret, self.track_window = cv2.CamShift(dst, self.track_window, CamShiftTracking.TERM_CRITERIA)
            (x, y, w, h) = (int(ret[0][0]), int(ret[0][1]), int(ret[1][0]), int(ret[1][1]))
        # expanded bird tracked size, and draw area
        # cv2.rectangle(bird_frame, (x - 10, y - 10), (x + w + 5, y + h - 5), (0, 255, 0), cv2.FILLED)
        self.result_area = (x - 12, y - 12), (x + w + 7, y + h - 7)
        return result.DiscretizationResult(self.result_area)

    def track_areas(self, frame):
        """
        :param frame: a current image frame
        :return: an area tracked by the Camshift algorithm
        """
        self.current_frame = frame
        if self._frame_count == 0:
            self._result = self.__init_track()
        else:
            self._result = self.__track_area()

        if self._result != None:
            self._frame_count = self._frame_count + 1
        else:
            raise Exception("no result has been calculated")

        return self._result
