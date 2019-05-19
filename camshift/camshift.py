import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy

reft = []

def selectROI(event, x, y, flags, param):
    # grab references to the global variables
    global refPt

    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        #cropping = True

    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        #cropping = False

        # draw a rectangle around the region of interest
        cv2.rectangle(frame, refPt[0], refPt[1], (0, 255, 0), 2)
        cv2.imshow("bird", frame)

cv2.namedWindow("bird")
cv2.setMouseCallback("bird", selectROI)

#init game and get first frame
game = flappy.GameState()
frame = game.next_frame(False)

#show the image and get the draw area
cv2.imshow('bird', frame)
key = cv2.waitKey(0) & 0xFF

c, w, r, h = refPt[0][0], refPt[1][0] - refPt[0][0], refPt[0][1], refPt[1][1] - refPt[0][1]
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)

mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
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
        frame = game.next_frame(True)
    if k == ord('s'):
        run = False
        frame = game.next_frame(False)
    if k == 32:
        flap = True
    if run:
        frame = game.next_frame(flap)


    ret = True
    if run:
        if track_window:
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

            # apply CamShift to get the new location
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            # Draw it on image
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            result = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('BackProjection', dst)
            cv2.imshow("bird",result)
        else:
            cv2.imshow("bird",frame)

game.quit()
cv2.destroyAllWindows()

