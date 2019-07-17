import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import gameAI.camshift as camshift
import gameAI.template_contour as tracker
import gameAI.sensor as sr

# init game and get first frame
game = flappy.GameState(0)

game_frame = game.next_frame(False)

# camshift = camshift.CamShiftTracking()
tracker = tracker.TemplateContour()
run = True
flap = False

cv2.imshow("GameAIVision", game_frame)
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


    # copy for contours tracking
    contours_frame = np.copy(game_frame)
    # copy for template tracking
    template_frame = np.copy(game_frame)

    ret = True
    tracker.width = game_frame.shape[0]
    tracker.height = game_frame.shape[1]
    # copy of game frame for bird tracking
    img = np.zeros((    tracker.width,tracker.height, 3), np.float)
    if run:
        discRes = tracker.track_areas(game_frame)
        # img = discRes.getAreaImage(img)
        walls = discRes.getGameWalls(game_frame.shape[0], game_frame.shape[1])
        for wall in walls:
            wall.draw(img)
        birdFrontCenter = discRes.getBirdFrontCenter()
        sensor1 = sr.Sensor(birdFrontCenter.x, birdFrontCenter.y, 45)
        sensor2 = sr.Sensor(birdFrontCenter.x, birdFrontCenter.y, 0)
        sensor3 = sr.Sensor(birdFrontCenter.x, birdFrontCenter.y, -45)
        cv2.rectangle(img,discRes.birdArea.leftTop.toIntTuple(),discRes.birdArea.rightDown.toIntTuple(),[255]*3)
        sensor1.castAndDraw(walls, img)
        sensor2.castAndDraw(walls, img)
        sensor3.castAndDraw(walls, img)
        cv2.imshow('GameAIVision', img)

game.quit()
cv2.destroyAllWindows()
