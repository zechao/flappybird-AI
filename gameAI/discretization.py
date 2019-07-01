import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import gameAI.camshift as camshift
import gameAI.template_contour as tracker
import gameAI.config as config

# init game and get first frame
game = flappy.GameState(0)

game_frame = game.next_frame(False)

# camshift = camshift.CamShiftTracking()
tracker = tracker.TemplateContour()
run = True
flap = False

while True:
    flap = False
    k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
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
        result_area = tracker.track_areas(game_frame)
        bird_frame = result_area.getAreaImage(bird_frame, game_frame.shape[0], game_frame.shape[1])
        cv2.imshow('Flappy Bird', bird_frame)

game.quit()
cv2.destroyAllWindows()
