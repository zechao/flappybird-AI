import numpy as np
import gameAI.ANN.activation_funtion as af


# class NeuralNet ():
#     # def __init__(self, inputNode, hiddenNode, outputNode, hiddenLayers):
#     #     if isinstance(inputNode, list) or isinstance(inputNode, tuple):
#     #         self.input = len(inputNode) + 1  # extra for bias node
#     #     else:
#     #         self.input = inputNode + 1
#     #     self.hNodes = hiddenNode
#     #     self.oNodes = outputNode
#     #     self.hiddenLayers = hiddenLayers
#     #
#     #     # # set up array of 1s for activations
#     #     # self.ai = [1.0] * self.inputNode
#     #     # self.ah = [1.0] * self.hidden
#     #     # self.ao = [1.0] * self.output
#     #     # # create randomized weights
#     #     # self.wi = np.random.randn(self.inputNode, self.hidden)
#     #     # self.wo = np.random.randn(self.hidden, self.output)
#     #     # # create arrays of 0 for changes
#     #     # self.ci = np.zeros((self.inputNode, self.hidden))
#     #     # self.co = np.zeros((self.hidden, self.output))


class NeuralLayer():
    def __init__(self, input, output, actFunction):
        """
        :param input: number of input
        :param output: number of output
        :param actFunction activation function
        """
        self.nInput = input
        self.nOutput = output
        self.f = actFunction
        # init bias value
        self.bias = np.random.rand(1, output)
        # init matrix of weight
        self.weight = np.random.rand(self.nInput, self.nOutput)

    def feed_forward(self, X):
        # Xi*Wij + bj
        z = np.dot(X, self.weight) + self.bias
        y = self.f(z)
        return y


if __name__ == '__main__':
    np.random.seed(1)
    layer = NeuralLayer(3, 2, af.sigmoid)
    r = layer.feed_forward(np.array([[1, 2, 3]]))
    print(r)
