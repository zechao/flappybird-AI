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
        return

    def computeInput(self, action):
        if self.game.crash:
            self.die = False
        game_frame = self.game.next_frame(action)
        game_frame = np.copy(game_frame)
        self.discRes = self.tracker.track_areas(game_frame)

        self.walls = self.discRes.getGameWalls(flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight())
        self.birdFrontCenter = self.discRes.getBirdFrontCenter()

        self.inputs = []
        for sensor in self.sensors:
            res = sensor.customPositionCast(self.birdFrontCenter.x, self.birdFrontCenter.y, self.walls)
            self.inputs.append(res)
        return False

    def draw(self, img):
        img = self.discRes.getAreaImage(img)
        cv2.rectangle(img, self.discRes.birdArea.leftTop.toIntTuple(), self.discRes.birdArea.rightDown.toIntTuple(),
                      [255] * 3)
        for wall in self.walls:
            wall.draw(img)

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

    def computeOutput(self):
        input = self._covertToNetInput()
        outPut = self.neuralNet.feed_forward(input)
        self.output = outPut
        return outPut

    def getActionFromOutput(self):
        if self.output > 0.6:
            return True
        else:
            return False


if __name__ == '__main__':
    net = nn.NeuralNet.createRandomNeuralNet(4, 6, 1, 2, actFunction=af.relu)
    ai = FlappyBirdAI([45, -45], net)
    ai.restAndRun()

    action = False
    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
    crash = ai.computeInput(action)
    ai.draw(img)
    bestFitness = 50
    bestNet = None
    repeatCount = 0
    while True:
        if not ai.die:
            cv2.imshow('GameAIVision', img)
            ai.computeOutput()
            action = ai.getActionFromOutput()
            img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
            crash = ai.computeInput(action)
            ai.draw(img)
            cv2.putText(img, 'Best fitness:' + str(ai.getFitness()), (20, 500), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                        [0, 0, 255],
                        1)
            k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
        else:
            cv2.destroyAllWindows()
            if ai.getFitness() > bestFitness:
                repeatCount = 0
                bestFitness = ai.getFitness()
                bestNet = ai.neuralNet.clone()
                ai.neuralNet.mutate(0.05)
            elif ai.getFitness() == bestFitness:
                repeatCount += 1
                ai.neuralNet.mutate(0.05 * repeatCount)
            elif ai.getFitness() < bestFitness:
                repeatCount += 1
                ai.neuralNet.mutate(0.05 * repeatCount)
            else:
                ai.neuralNet.mutate(0.5)

            ai.game.resetAndRun()
