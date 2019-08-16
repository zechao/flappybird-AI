import numpy as np
import gameAI.ANN.activation_funtion as af
import pickle


def buildVecFunc(mutateRate, mutateFunc):
    """
    :param mutateRate:
    :param mutateFunc:
    :return:  func for mutate element if  of numpy.array
    """

    def vFunc(x):
        threshold = np.random.random()
        if threshold < mutateRate:
            return mutateFunc(x)
        else:
            return x

    return np.vectorize(vFunc)


def defaultMutateFunc(x):
    x += np.random.uniform(-0.2, 0.2)
    return x


class NeuralLayer():
    def __init__(self, inputNum, outputNum, actFunction=af.sigmoid, **kw):
        """
        :param input: number of input
        :param output: number of output
        :param actFunction activation function
        """
        self.iNode = inputNum
        self.oNode = outputNum
        self.f = actFunction

        # init bias if value
        if 'bias' in kw:
            self.hasBias = True
            self.bias = kw['bias']
        else:
            self.hasBias = False
            self.bias = np.random.random()

        # init matrix of weight
        self.weight = (2 * np.random.random_sample((self.iNode, self.oNode)) - 1)

    def feed_forward(self, X):
        """
        Calc the result matrix with given input
        :param X: 1*inputNum matrix for the input
        :return: the result matrix with the current weight configuration
        """
        # Xi*Wij + bj
        z = np.dot(X, self.weight) + self.bias
        y = self.f(z)
        return y

    def mutate(self, mutateRate, mutateFunc=defaultMutateFunc):
        """

        :param mutateRate: probability of mutation [0.0,1.0]
        :param mutateFunc: mutation logic, default use defaultMutateFunc
        :return:
        """
        vecFunc = buildVecFunc(mutateRate, mutateFunc)
        self.weight = vecFunc(self.weight)
        self.mutateBias()

    def mutateBias(self):
        self.bias += np.random.uniform(-1, 1)

    def crossover(self, otherLayer, ):
        if isinstance(otherLayer, NeuralLayer):
            return self.clone()
        else:
            child = self.clone()
            [rows, cols] = child.weight.shape
            for i in range(rows):
                for j in range(cols):
                    if np.random.random() <= 0.3:
                        child.weight[i, j] = otherLayer[i, j]
                        child.bias = otherLayer.bias

    def clone(self):
        layer = NeuralLayer.createCustomLayer(self.iNode, self.weight, actFunction=self.f, bias=self.bias)
        return layer

    @staticmethod
    def createCustomLayer(inputNum, initWeight, actFunction=af.sigmoid, **kw):
        """
        Create random layer init a layer with specific weight
        :param inputNum: number of input node
        :param initWeight: numpy matrix of layer weight, the dimension must coincide with the input
        :param actFunction:
        :param kw: for init bias
        :return:
        """
        neural = NeuralLayer(inputNum, initWeight.shape[1], actFunction, **kw)
        neural.weight = initWeight.copy()
        return neural

    @staticmethod
    def createRandomLayer(inputNode, outPutNode, actFunction=af.sigmoid):
        """
        Create random layer init a layer with specific inputNum and outPut
        :param inputNode: number of input node
        :param outPutNode:  number of output node
        :param actFunction: default activation function is sigmoid
        :return:
        """
        return NeuralLayer(inputNode, outPutNode)

    def __str__(self):
        return format(self.weight)

    def __eq__(self, other):
        if other == None:
            return False
        return np.array_equal(self.weight, other.weight)


class NeuralNet():
    def __init__(self, layers, **kwargs):
        if not isinstance(layers, list):
            raise TypeError("list of layers required")
        # check if each layer is correct connect, input and output of middle layers must match
        layerN = len(layers)
        if layerN < 1:
            raise RuntimeError("no layers")
        # only one layer for the net

        self.iNode = layers[0].iNode
        self.oNode = layers[-1].oNode
        if layerN > 1:
            self.hLayer = layerN - 1
        # check if layer input and output is correct
        for i in range(len(layers) - 1):
            if layers[i].oNode != layers[i + 1].iNode:
                raise RuntimeError("layers input and output not matching")
        self.layers = layers

    def feed_forward(self, X):
        input = X
        for eachLayer in self.layers:
            input = eachLayer.feed_forward(input)
        return input

    def crossover(self, other):
        child = self.clone()
        for i in range(len(self.layers)):
            child.layers[i] = self.layers[i].crossover(other.layers[i])
        return child

    def mutate(self, mutateRate):
        for each in self.layers:
            each.mutate(mutateRate)

    @staticmethod
    def createRandomNeuralNet(inputNum, hiddenNum, outputNum, hiddenLayerNum=1, actFunction=af.sigmoid,
                              outputActFunc=None, **kw):
        # the neural net has only one layer
        if hiddenLayerNum == 0:
            layer = NeuralLayer(inputNum, outputNum, actFunction=actFunction)
            return NeuralNet([layer])
        else:
            layers = []
            layers.append(NeuralLayer(inputNum, hiddenNum, actFunction=actFunction))
            for x in range(1, hiddenLayerNum):
                layers.append(NeuralLayer(hiddenNum, hiddenNum, actFunction=actFunction))
            if outputActFunc == None:
                layers.append(NeuralLayer(hiddenNum, outputNum, actFunction=actFunction))
            else:
                layers.append(NeuralLayer(hiddenNum, outputNum, actFunction=outputActFunc))
        return NeuralNet(layers)

    def clone(self):
        layers = []
        for eachLayer in self.layers:
            layers.append(eachLayer.clone())
        return NeuralNet(layers)

    def __str__(self):
        s = ""
        for i, eachLayer in enumerate(self.layers):
            s += 'layer ' + str(i + 1) + "\t"
            s += format(eachLayer) + '\n'
        return s

    def __eq__(self, other):
        if other == None:
            return False
        if len(self.layers) != len(self.layers):
            return False
        for idx in range(len(self.layers)):
            if self.layers[idx] != other.layers[idx]:
                return False
        return True

