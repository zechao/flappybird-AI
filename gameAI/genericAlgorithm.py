import gameAI.flappyBirdAI as ai
import gameAI.ANN.neuralnetwork as nn
import gameAI.ANN.activation_funtion as af
import game.wrapped_flappy_bird as flappy
import cv2
import numpy as np


class GenericAlgorithm():

    def __init__(self, max_population, top_units):
        self.max_population = max_population
        self.top_units = top_units
        self.currentIterCount = 1
        self.initMutateRate = 0.1  # [0.0,1.0]
        self.bestFitness = 0
        self.bestScore = 0

    def initPopulation(self):
        self.population = []
        self.winner = []
        for x in range(self.max_population):
            net = nn.NeuralNet.createRandomNeuralNet(6, 6, 1, 2, actFunction=af.relu)
            # create each AI and begin
            eachAI = ai.FlappyBirdAI([90, -90, 0], net)
            eachAI.restAndRun()
            self.population.append(eachAI)

    def activeAll(self):
        for bird in self.population:
            if not bird.die:
                outPut = bird.activateNet()
                if bird.output > 0.6:
                    bird.flapWings(True)
                else:
                    bird.flapWings(False)

    def draw(self, img):
        wallDraw = True
        for bird in self.population:
            if not bird.die:
                bird.drawBird(img)
                bird.drawSensor(img)
                if wallDraw:
                    bird.drawWalls(img)
                    wallDraw = False

    def determineNextAction(self):
        for bird in self.population:
            if not bird.die:
                bird.determineNextAction()

    def computeInput(self):
        for bird in self.population:
            if not bird.die:
                bird.computeInput()

    def areAllDied(self):
        count = 0
        for bird in self.population:
            if bird.die:
                count += 1
        return count == self.max_population


    def selection(self):
        self.population.sort(key=lambda bird:bird.game.fitness,reverse=True)
        #select the best
        self.best = self.population[0]
        self.bestFitness = self.best.getFitness()
        self.bestScore = self.best.getScore()

    def restAndRun(self):
        self.lifeCount = len(self.population)
        for bird in self.population:
            bird.restAndRun()

    def mutate(self):
        for bird in self.population:
            bird.neuralNet.mutate(0.8)


if __name__ == '__main__':
    generic = GenericAlgorithm(10, 3)
    generic.initPopulation()

    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
    generic.draw(img)
    while True:
        cv2.imshow('GameAIVision', img)
        k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
        if not generic.areAllDied():
            generic.computeInput()
            generic.activeAll()
            generic.determineNextAction()
            img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
            generic.draw(img)
        else:
            generic.generic()
            cv2.destroyAllWindows()
            generic.mutate()
            generic.restAndRun()
