import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import gameAI.discretization.area_finder as af
import gameAI.discretization.sensor as sr

# init game and get first frame



areaFinder = af.AreaFinder()
imgName = 'frame1000.png'
while True:
    cv2.waitKey(0)
    game_frame = cv2.imread(imgName)
    game_frame = cv2.cvtColor(game_frame, cv2.COLOR_BGR2RGB)
    sensors = [
        sr.Sensor.createBlankSensor(45),
        sr.Sensor.createBlankSensor(0),
        sr.Sensor.createBlankSensor(-45),
        sr.Sensor.createBlankSensor(90),
        sr.Sensor.createBlankSensor(-90),
        sr.Sensor.createBlankSensor(-30),
        sr.Sensor.createBlankSensor(30),
        sr.Sensor.createBlankSensor(-15),

        sr.Sensor.createBlankSensor(15),
    ]

    ret = True
    # copy of game frame for bird tracking
    img = np.zeros((flappy.getCV2ScreenWidth()+10, flappy.getCV2ScreenHeight()+10, 3), np.float)
    discRes = areaFinder.track_areas(game_frame)
    # img = discRes.getAreaImage(img)
    walls = discRes.getGameWalls(flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight())
    for wall in walls:
        wall.draw(img)
    birdFrontCenter = discRes.getBirdFrontCenter()

    cv2.rectangle(img, discRes.birdArea.leftTop.toIntTuple(), discRes.birdArea.rightDown.toIntTuple(), [255,0,0])
    for sensor in sensors:
        sensor.customPositionCast(birdFrontCenter.x, birdFrontCenter.y, walls, img)

    cv2.imshow("distance",img)

cv2.destroyAllWindows()
