import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import os


def draw_bird_tracked_area(wname, img, ret):
    # Draw it on image_test
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    result = cv2.polylines(img, [pts], True, 255, 2)
    cv2.imshow(wname, result)


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


def get_bird_body_area():
    return np.array((70, 5, 248, 10))


def isOverLapping(area1, area2):
    r1x = area1[0]
    r1y = area1[1]
    r1width = area1[2]
    r1height = area1[3]

    r2x = area2[0]
    r2y = area2[1]
    r2width = area2[2]
    r2height = area2[3]


    return not(r1x + r1width < r2x or
             r1y + r1height < r2y or
             r1x > r2x + r2width or
             r1y > r2y + r2height)


cv2.namedWindow("bird")
# init game and get first frame
game = flappy.GameState()
game_frame = game.next_frame(False)

# select area of bird, hardcoded!
c, w, r, h = get_bird_body_area()
cv2.rectangle(game_frame, (c, r), (c + w, r + h), (0, 255, 0), 1)

# show the image_test and get the draw area
cv2.imshow('bird', game_frame)
# tracking area used for CameShift
track_window = (c, r, w, h)
# set up the ROI for tracking

roi = game_frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

# mask area for bird body color
mask = cv2.inRange(hsv_roi, get_bird_hsv_low(), get_bird_hsv_high())
# calc histogram for bird body mask
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
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
    tamplate_frame = np.copy(game_frame)

    ret = True
    if run:
        hsv = cv2.cvtColor(bird_frame, cv2.COLOR_RGB2HSV)

        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
        # apply CamShift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        # expanded b  ird tracked size, and draw area
        (x, y, w, h) = (int(ret[0][0]), int(ret[0][1]), int(ret[1][0]), int(ret[1][1]))
        cv2.rectangle(bird_frame, (x - 10, y - 10), (x + w + 5, y + h - 5), (0, 255, 0), cv2.FILLED)
        draw_bird_tracked_area("bird", bird_frame, ret)
        # draw_bird_backProject("BackProject", dst)

        # Track bird with template matching
        img_gray = cv2.cvtColor(tamplate_frame, cv2.COLOR_BGR2GRAY)
        template = cv2.imread('redbird-downflap.png', 0)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(tamplate_frame, top_left, bottom_right, (255, 0, 0), cv2.FILLED)

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
                if isOverLapping((top_left[0], top_left[1], w, h),(cx, cy, cw, ch)):
                    cv2.rectangle(contours_frame, top_left, bottom_right, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.rectangle(contours_frame, (cx-2, cy-2), (cx + cw+4, cy + ch+4), (0, 0, 255), cv2.FILLED)

        cv2.imshow("tamplate_frame", tamplate_frame)
        cv2.imshow('contours', contours_frame)




    else:
        cv2.imshow("bird", game_frame)

game.quit()
cv2.destroyAllWindows()
