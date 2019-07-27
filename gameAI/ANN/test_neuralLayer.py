from unittest import TestCase
import numpy as np
import gameAI.ANN.activation_funtion as af
import gameAI.ANN.neuralnetwork as nn


class TestNeuralLayer(TestCase):

    def test_feed_forward(self):
        layer = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.5, 0.6],
                      [-0.1, 0.1],
                      [-0.2, 0.7]]),
            lambda x: x)
        r = layer.feed_forward(np.array([[1, -1, 2]]))

        layer = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.5, 0.6],
                      [-0.1, 0.1],
                      [-0.2, 0.7]]),
            af.sigmoid)
        r = layer.feed_forward(np.array([[1, -1, 2]]))
        self.assertTrue(self, np.array_equal(r, np.array([[0.549834, 0.86989153]])))

    def test_clone_equal(self):
        layer1 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.1, 0.2],
                      [-0.1, 0.1],
                      [-0.2, 0.1]]),
            af.sigmoid)

        layer2 = layer1.clone()
        self.assertEqual(layer1, layer2)
        layer2.mutate(0.5)
        self.assertNotEqual(layer1, layer2)

    def test_mutate_clone(self):
        layer1 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.1, 0.2],
                      [-0.1, 0.1],
                      [-0.2, 0.1]]),
            af.sigmoid)
        layer2 = layer1.clone()

        layer2.mutate(0.5)
        self.assertNotEqual(layer1, layer2)

    def test_crossover_and_mutate(self):
        np.random.seed(1)
        layer1 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.5, 0.6],
                      [-0.1, 0.1],
                      [-0.2, 0.7]]),
            af.sigmoid)
        layer2 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.1, 0.2],
                      [-0.1, 0.1],
                      [-0.2, 0.1]]),
            af.sigmoid)
        child = layer1.crossover(layer2)
        print(child)
        child.mutate(0.2)
        print("mutate child", child)


class TestNeuralNet(TestCase):

    def test_clone_equal(self):
        net1 = nn.NeuralNet.createRandomNeuralNet(5, 6, 1, 10)
        net2 = net1.clone()
        self.assertEqual(net1, net2)
        net2.mutate(0.5)
        self.assertNotEqual(net1, net2)
