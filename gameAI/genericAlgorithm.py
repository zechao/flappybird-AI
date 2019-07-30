import gameAI.flappyBirdAI as ai
import gameAI.ANN.neuralnetwork as nn
import gameAI.ANN.activation_funtion as af


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
            net = nn.NeuralNet.createRandomNeuralNet(4, 6, 1, 2, actFunction=af.relu)
            #create each AI and begin
            eachAI = ai.FlappyBirdAI([45, -45], net)
            eachAI.restAndRun()

    def activeAll(self):
        for each in self.population:
            if not each.die:
                each.computeInput(False)
