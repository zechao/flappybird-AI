import numpy as np
import gameAI.ANN.activation_funtion as af


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
    x += np.random.normal() / 5
    if x > 1:
        return 1
    if x < -1:
        return -1
    else:
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
            self.bias = 0

        # init matrix of weight
        self.weight = np.random.rand(self.iNode, self.oNode)

    def feed_forward(self, X):
        # Xi*Wij + bj
        z = np.dot(X, self.weight) + self.bias
        y = self.f(z)
        return y

    def mutate(self, mutateRate, mutateFunc=defaultMutateFunc):
        vecFunc = buildVecFunc(mutateRate, mutateFunc)
        self.weight = vecFunc(self.weight)

    def crossover(self, otherLayer):
        if isinstance(otherLayer, NeuralLayer):
            return self.clone()
        else:
            child = self.clone()
            [rows, cols] = child.weight.shape
            print(rows, cols)
            for i in range(rows):
                for j in range(cols):
                    if np.random.random() <= 0.5:
                        child.weight[i, j] = otherLayer[i, j]

    def clone(self):
        layer = NeuralLayer.createCustomLayer(self.iNode, self.weight, actFunction=self.f)
        return layer

    @staticmethod
    def createCustomLayer(inputNum, initWeight, actFunction=af.sigmoid, **kw):
        neural = NeuralLayer(inputNum, initWeight.shape[1], actFunction, **kw)
        neural.weight = initWeight.copy()
        return neural

    @staticmethod
    def createRandomLayer(inputNum, output, actFunction=af.sigmoid):
        return NeuralLayer(inputNum, output)

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

    def crossover(self, other):
        child = self.clone()
        for i in range(len(self.layers)):
            child.layers[i] = self.layer[i].crossover(other.layers[i])
        return child

    def mutate(self, mutateRate):
        for each in self.layers:
            each.mutate(mutateRate)

    @staticmethod
    def createRandomNeuralNet(inputNum, hiddenNum, outputNum, hiddenLayerNum=5, actFunction=af.sigmoid, **kw):
        # the neural net has only one layer
        if hiddenLayerNum == 0:
            layer = NeuralLayer(inputNum, outputNum, actFunction=actFunction)
            return NeuralNet([layer])
        else:
            layers = []
            layers.append(NeuralLayer(inputNum, hiddenNum, actFunction=actFunction))
            for x in range(1, hiddenLayerNum):
                layers.append(NeuralLayer(hiddenNum, hiddenNum, actFunction=actFunction))
            layers.append(NeuralLayer(hiddenNum, outputNum, actFunction=actFunction))
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


if __name__ == '__main__':
    pass
