import pickle


def saveNet(filePath, neuralNet):
    with open(filePath, "wb") as f:
        pickle.dump(neuralNet, f)


def loadNet(filePath):
    with open(filePath, "rb") as f:
        net = pickle.load(f)
        return net
