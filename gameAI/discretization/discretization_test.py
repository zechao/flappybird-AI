import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import gameAI.discretization.area_finder as af
import gameAI.discretization.sensor as sr

# init game and get first frame
game = flappy.GameState(0)


areaFinder = af.AreaFinder()
run = True
flap = False
game_frame = None
# cv2.imshow("GameAIVision", game_frame)
img = None
die = False
game.resetAndRun()
while True and not game.crash:
    flap = False
    k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
    if die:
        cv2.imshow('GameAIVision', img)
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
        game_frame = game.next_frame(True)
    if k == 32:
        flap = True
    if run:
        game_frame = game.next_frame(True)

    # # copy for pipeContours tracking
    # contours_frame = np.copy(game_frame)
    # # copy for template tracking
    # template_frame = np.copy(game_frame)
    sensors = [
        sr.Sensor.createBlankSensor(45),
        sr.Sensor.createBlankSensor(0),
        sr.Sensor.createBlankSensor(-45),
    ]
    ret = True
    # copy of game frame for bird tracking
    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight()+200, 3), np.float)
    if run:
        discRes = areaFinder.track_areas(game_frame)
        # img = discRes.getAreaImage(img)
        walls = discRes.getGameWalls(flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight())
        for wall in walls:
            wall.draw(img)
        birdFrontCenter = discRes.getBirdFrontCenter()

        cv2.rectangle(img, discRes.birdArea.leftTop.toIntTuple(), discRes.birdArea.rightDown.toIntTuple(), [255] * 3)
        for sensor in sensors:
            sensor.customPositionCast(birdFrontCenter.x, birdFrontCenter.y, walls, img)
        cv2.imshow('GameAIVision', img)

game.quit()
cv2.destroyAllWindows()
