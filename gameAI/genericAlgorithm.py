import gameAI.flappyBirdAI as ai
import gameAI.ANN.neuralnetwork as nn
import gameAI.ANN.activation_funtion as af
import game.wrapped_flappy_bird as flappy
import cv2
import numpy as np
import random
import time
from multiprocessing import Pool


class GenericAlgorithm():

    def __init__(self, max_population, top_units):
        self.max_population = max_population
        self.top_units = top_units
        self.currentIterCount = 1
        self.initMutateRate = 0.1
        self.bestFitness = 0
        self.lastBestFitness = 0
        self.bestScore = 0
        self.bestFitnessUnchangedCount = 0

    def initPopulation(self, angles, inputNum, hiddenNum, outputNum, hiddenLayerNum, actFunction=af.sigmoid,
                       outputActFunc=None, **kw):
        self.generationCount = 0
        self.population = []
        self.winner = []
        for x in range(self.max_population):
            self.net = nn.NeuralNet.createRandomNeuralNet(inputNum, hiddenNum, outputNum, hiddenLayerNum, actFunction,
                                                          outputActFunc)
            # create each AI and begin
            eachAI = ai.FlappyBirdAI(angles, self.net)
            eachAI.restAndRun()
            self.population.append(eachAI)

    def activeAll(self):
        for bird in self.population:
            if not bird.die:
                outPut = bird.activateNet()
                if bird.output[0, 0] > bird.output[0, 1]:
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
        """
        select best without doing anything
        :return:
        """
        self.generationCount += 1
        # sort desc with fitness
        self.population.sort(key=lambda bird: bird.game.fitness, reverse=True)
        # select the best top units
        self.best = self.population[0]
        self.bestFitness = self.best.getFitness()
        self.bestScore = self.best.getScore()

        if self.lastBestFitness == self.bestFitness:
            self.bestFitnessUnchangedCount += 1

        print("Generation:{}->Best with score:{}, with fitness:{}".format(self.generationCount, self.bestScore,
                                                                          self.bestFitness))

        # select the top for the crossover and mutation
        self.interMediatePopulation = self.population[:self.top_units]

        # the best will store directly to ensure the best result
        self.population = [self.best]
        self.lastBestFitness = self.bestFitness

    def restAndRun(self):
        self.lifeCount = len(self.population)
        for bird in self.population:
            bird.restAndRun()

    def selectParent(self):
        idx = random.randint(0, self.top_units - 1)
        return self.interMediatePopulation[idx].clone()

    def crossover(self):
        while len(self.population) < self.max_population:
            parent1 = self.selectParent()
            parent2 = self.selectParent()
            parent1.crossover(parent2)
            self.population.append(parent1)

    def mutate(self, mutationRate, maxMutationRate):
        for bird in self.population[1:]:
            if self.bestFitnessUnchangedCount == 5:
                bird.neuralNet.mutate(np.random.uniform(mutationRate, maxMutationRate))
                self.bestFitnessUnchangedCount = 0
            else:
                bird.neuralNet.mutate(mutationRate)


if __name__ == '__main__':
    np.random.seed(0)
    generic = GenericAlgorithm(20, 3)
    generic.initPopulation([80, -80, -45, 45,0], 10, 20, 2, 1, actFunction=af.tanh, outputActFunc=af.tanh)

    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
    generic.draw(img)
    while True:
        cv2.imshow('GameAIVision', img)
        k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
        if not generic.areAllDied():
            generic.computeInput()
            generic.activeAll()
            # generic.determineNextAction()
            img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight(), 3), np.float)
            start = time.time()
            generic.draw(img)
        else:
            cv2.destroyAllWindows()
            generic.selection()
            generic.crossover()
            generic.mutate(0.2, 0.6)
            generic.restAndRun()
