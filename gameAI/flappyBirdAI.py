import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import time
import gameAI.template_contour as tracker
import gameAI.sensor as sr
import gameAI.ANN.neuralnetwork as nn


# init game and get first frame


class FlappyBirdAI():

    def __init__(self, sensors, neuralNet):
        self.game = flappy.GameState(0)
        self.tracker = tracker.TemplateContour()
        self.sensors = sensors
        self.neuralNet = neuralNet

    def getFitness(self):
        return self.game.fitness

    def gameRunning(self):
        return self.game.running

    def restAndRun(self):
        return self.game.resetAndRun()

    def computeInput(self, action, img=None):
        if self.game.crash:
            return True
        game_frame = self.game.next_frame(action)
        game_frame = np.copy(game_frame)
        self.discRes = self.tracker.track_areas(game_frame)
        # img = discRes.getAreaImage(img)
        self.walls = self.discRes.getGameWalls(flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight())

        self.birdFrontCenter = self.discRes.getBirdFrontCenter()

        self.inputs = []
        for sensor in self.sensors:
            res = sensor.customPositionCastAndDraw(self.birdFrontCenter.x, self.birdFrontCenter.y, walls, img)
            self.inputs.append(res)
        return False

    def draw(self,img):
        if img is not None:
            cv2.rectangle(img, self.discRes.birdArea.leftTop.toIntTuple(), self.discRes.birdArea.rightDown.toIntTuple(),
                          [255] * 3)
            for wall in self.walls:
                wall.draw(img)
        for sensor in self.sensors:
            res = sensor.customPositionCastAndDraw(self.birdFrontCenter.x, self.birdFrontCenter.y, walls, img)

    def _covertToNetInput(self):
        netInputs = []
        for each in self.inputs:
            netInputs.append(each.distance)

        return np.array([netInputs])

    def computeOutput(self):
        input = self._covertToNetInput()
        outPut = self.neuralNet.feed_forward(input)
        self.output = outPut/2
        return outPut

    def getActionFromOutput(self):
        if self.output > 0.5:
            return True
        else:
            return False


def createGameInstances(num=10):
    sensor = [
        sr.Sensor.createBlankSensor(-45),
        sr.Sensor.createBlankSensor(45),
        sr.Sensor.createBlankSensor(0),
    ]


    ais =[]
    for x in range(num):
        net = nn.NeuralNet.createRandomNeuralNet(3, 5, 1, 5)
        ais.append(FlappyBirdAI(sensor, net))

    for ai in ais:
        ai.restAndRun()

    action = False
    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
    for ai in ais:
        ai.computeInput(action, img)

    while True:
        if ai.game.isRunning():
            cv2.imshow('GameAIVision', img)
            ai.computeOutput()
            action = ai.getActionFromOutput()
            img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
            crash = ai.computeInput(action, img)
            cv2.putText(img, 'Best fitness:' + str(ai.getFitness()), (20, 500), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        [0, 0, 255],
                        1)
            k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
        else:
            cv2.destroyAllWindows()
            ai.neuralNet.mutate(0.3)
            ai.game.resetAndRun()



if __name__ == '__main__':
    pass
    sensor = [
        sr.Sensor.createBlankSensor(-45),
        # sr.Sensor.createBlankSensor(90),
        sr.Sensor.createBlankSensor(45),
        sr.Sensor.createBlankSensor(0),
    ]

    net = nn.NeuralNet.createRandomNeuralNet(3, 5, 1, 5)
    ai = FlappyBirdAI(sensor, net)
    ai.restAndRun()

    action = False
    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
    crash = ai.computeInput(action, img)

    while True:
        if ai.game.isRunning():
            cv2.imshow('GameAIVision', img)
            ai.computeOutput()
            action = ai.getActionFromOutput()
            img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
            crash = ai.computeInput(action, img)
            cv2.putText(img, 'Best fitness:' + str(ai.getFitness()), (20, 500), cv2.FONT_HERSHEY_COMPLEX, 0.6, [0, 0, 255],
                        1)
            k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
        else:
            cv2.destroyAllWindows()
            ai.neuralNet.mutate(0.3)
            ai.game.resetAndRun()

