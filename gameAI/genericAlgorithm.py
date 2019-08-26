import gameAI.flappyBirdAI as ai
import gameAI.ANN.neuralnetwork as nn
import gameAI.ANN.activation_funtion as af
import game.wrapped_flappy_bird as flappy
import gameAI.trainData as td
import cv2
import numpy as np
import random
import time
import os


class GenericAlgorithm():

    def __init__(self, max_population, top_units):
        self.max_population = max_population
        self.top_units = top_units
        self.bestFitness = 0
        self.lastBestFitness = 0
        self.bestScore = 0
        self.aliveCount = max_population
        self.startTime = time.strftime("%Y%M%d_%H_%M_%S")

    def initPopulation(self, angles, inputNum, hiddenNum, outputNum, hiddenLayerNum, actFunction=af.sigmoid,
                       outputActFunc=None, **kw):
        self.generationCount = 1
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
        self.aliveCount = 0
        self.population.sort(key=lambda bird: bird.game.fitness, reverse=True)
        self.bestAgent = self.population[0]
        self.bestFitness = self.population[0].getFitness()
        self.bestScore = self.population[0].getScore()
        for bird in self.population:
            if not bird.die:
                self.aliveCount += 1
                outPut = bird.activateNet()
                if bird.output[0, 0] > bird.output[0, 1]:
                    bird.flapWings(True)
                else:
                    bird.flapWings(False)



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

    def restAndRun(self):
        self.lifeCount = len(self.population)
        for bird in self.population:
            bird.restAndRun()

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

        print("Generation:{}->Best with score:{}, with fitness:{}".format(self.generationCount, self.bestScore,
                                                                          self.bestFitness))

        # select the top for the crossover and mutation
        self.interMediatePopulation = self.population[:self.top_units]

        # the best will store directly to ensure the best result
        self.population = [self.best]
        # add another an clone of the best one for mutate

        self.population.append(self.best.clone())
        self.population.append(self.best.clone())
        self.lastBestFitness = self.bestFitness

        # compute average sum fitness of interMediatePopulation in order to compute the probability of selection
        self.sumInterMediatePopulation = 0
        for each in self.interMediatePopulation:
            self.sumInterMediatePopulation += each.getFitness()
        print("total fitness:", self.sumInterMediatePopulation)

    def selectParent(self):
        idx = random.randint(0, self.top_units - 1)
        percentage = np.random.random()
        for candidate in self.interMediatePopulation:
            percentage -= candidate.getFitness() / self.sumInterMediatePopulation
            if percentage <= 0:
                return candidate.clone()
        return self.interMediatePopulation[-1].clone()

    def crossover(self):
        while len(self.population) < self.max_population:
            parent1 = self.selectParent()
            parent2 = self.selectParent()
            parent1.crossover(parent2)
            self.population.append(parent1)

    def mutate(self, mutationRate):
        for bird in self.population[1:]:
            bird.neuralNet.mutate(mutationRate)

    def drawGenerationInfo(self,img):
        self.drawGeneration(img)
        self.drawAlive(img)
        self.drawBestFitness(img)
        self.drawBestScore(img)

    def drawGeneration(self, img):
        cv2.putText(img, 'Generation:' + str(self.generationCount), (300, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    [0, 0, 255],
                    1)

    def draw(self, img):
        wallDraw = True
        for bird in self.population:
            if not bird.die:
                bird.drawBird(img)
                bird.drawSensor(img)
                if wallDraw:
                    bird.drawWalls(img)
                    wallDraw = False

    def drawAlive(self, img):
        cv2.putText(img, 'Alive:' + str(self.aliveCount), (300, 70), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    [0, 0, 255],
                    1)

    def drawBestFitness(self, img):
        cv2.putText(img, 'Best Fitness :' + str(self.bestFitness), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    [0, 0, 255],
                    1)

    def drawBestScore(self, img):
        cv2.putText(img, 'Best Score :' + str(self.bestScore), (300, 130), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                    [0, 0, 255],
                    1)

    def saveGenerationInfo(self):
        data = "{},{},{}\n".format(self.generationCount - 1, self.bestFitness, self.bestScore)
        directory= "data/train{}/".format(self.startTime)
        self.createDirectory(directory)
        with open('data/train{}/data.csv'.format(self.startTime), 'a') as f:
            f.write(data)

    def saveBest(self):
        directory = "data/train{}/".format(self.startTime)
        self.createDirectory(directory)
        td.saveNet("data/train{}/generation{}".format(self.startTime, self.generationCount-1), self.best.neuralNet)

    def createDirectory(self,directory):
        if not os.path.exists(os.path.dirname(directory)):
            os.makedirs(os.path.dirname(directory))


if __name__ == '__main__':
    np.random.seed(1)
    generic = GenericAlgorithm(20, 5)
    generic.initPopulation([-90, 90, 45, -45, 0], 10, 10, 2, 2, actFunction=af.relu, outputActFunc=af.relu)

    img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight() + 300, 3), np.float)
    generic.draw(img)
    generic.drawAlive(img)
    while True:
        cv2.imshow('GameAIVision', img)
        k = cv2.waitKey(1) & 0xFF  # when using 64bit machine
        if not generic.areAllDied():
            if generic.generationCount <=150:
                generic.computeInput()
                generic.activeAll()
                img = np.zeros((flappy.getCV2ScreenWidth(), flappy.getCV2ScreenHeight() + 300, 3), np.float)
                generic.draw(img)
                generic.drawGenerationInfo(img)
            else:
                np.random.seed(np.random.randint(100))
                generic = GenericAlgorithm(40, 5)
                generic.initPopulation([-90, 90, 45, -45, 0], np.random.randint(20), np.random.randint(20), 2, 2, actFunction=af.relu, outputActFunc=af.relu)
        else:
            cv2.destroyAllWindows()
            generic.selection()
            generic.crossover()
            generic.mutate(0.2)
            generic.saveGenerationInfo()
            generic.saveBest()
            generic.restAndRun()
