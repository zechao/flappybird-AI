import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import camshift.CamShift as camshift

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






# init game and get first frame
game = flappy.GameState()

game_frame = game.next_frame(False)

# select area of bird, hardcoded!

camshift = camshift.CamShiftTracking(game_frame)

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
