import cv2
import numpy as np
import game.wrapped_flappy_bird as flappy
import gameAI.discretization.template_contour as tracker
import gameAI.discretization.sensor as sr
import gameAI.ANN.neuralnetwork as nn
import gameAI.ANN.activation_funtion as af


# init game and get first frame


class FlappyBirdAI():

    def __init__(self, angles, neuralNet):
        self.game = flappy.GameState(0)
        self.tracker = tracker.TemplateContour()
        self.angles = angles
        self.sensors = []
        for angle in self.angles:
            self.sensors.append(sr.Sensor.createBlankSensor(angle))
        self.neuralNet = neuralNet
        self.die = False

    def getFitness(self):
        return self.game.fitness

    def getScore(self):
        return self.game.score

    def gameRunning(self):
        return self.game.running

    def restAndRun(self):
        self.game.resetAndRun()
        self.action = False
        self.die = False
        self.computeInput()
        return

    def computeInput(self):
        game_frame = self.game.next_frame(self.action)
        self.die = self.game.crash
        game_frame = np.copy(game_frame)
        self.discRes = self.tracker.track_areas(game_frame)
        self.walls = self.discRes.getGameWalls(flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight())
        self.birdFrontCenter = self.discRes.getBirdFrontCenter()
        self.inputs = []
        for sensor in self.sensors:
            res = sensor.customPositionCast(self.birdFrontCenter.x, self.birdFrontCenter.y, self.walls)
            self.inputs.append(res)

    def draw(self, img):
        # img = self.discRes.getAreaImage(img)
        self.drawWalls(img)
        self.drawBird(img)
        self.drawSensor(img)

    def drawWalls(self, img):
        for wall in self.walls:
            wall.draw(img)

    def drawBird(self, img):
        cv2.rectangle(img, self.discRes.birdArea.leftTop.toIntTuple(), self.discRes.birdArea.rightDown.toIntTuple(),
                      [255] * 3)

    def drawSensor(self, img):
        for res in self.inputs:
            res.draw(img)

    def _covertToNetInput(self):
        netInputs = []
        for each in self.inputs:
            if each.boundType.OBSTACLE:
                netInputs.append(each.distance)
                netInputs.append(0)
            else:
                netInputs.append(0)
                netInputs.append(each.distance)

        return np.array([netInputs])

    def activateNet(self):
        input = self._covertToNetInput()
        outPut = self.neuralNet.feed_forward(input)
        self.output = outPut
        return self.output

    def determineNextAction(self):
        if self.output > 0.6:
            self.action = True
        else:
            self.action = False

    def flapWings(self, flap):
        self.action = flap


if __name__ == '__main__':
    net = nn.NeuralNet.createRandomNeuralNet(4, 6, 1, 2, actFunction=af.relu)
    ai = FlappyBirdAI([45, -45], net)
    ai.restAndRun()

    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
    ai.draw(img)
    while True:
        if not ai.die:
            cv2.imshow('GameAIVision', img)
            img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
            ai.computeInput()
            ai.activateNet()
            ai.determineNextAction()
            ai.draw(img)
            # cv2.putText(img, 'Best fitness:' + str(ai.getFitness()), (20, 500), cv2.FONT_HERSHEY_COMPLEX, 0.6,
            #             [0, 0, 255],
            #             1)
            k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
        else:
            cv2.destroyAllWindows()
            ai.neuralNet.mutate(0.8)
            ai.restAndRun()
