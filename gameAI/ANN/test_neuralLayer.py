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
        expected = np.array([[0.2, 1.9]])
        self.assertTrue(np.allclose(r, expected))

        layer = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.5, 0.6],
                      [-0.1, 0.1],
                      [-0.2, 0.7]]),
            lambda x: x - 0.1)
        r = layer.feed_forward(np.array([[1, -1, 2]]))
        expected = np.array([[0.1, 1.8]])
        self.assertTrue(np.allclose(r, expected))

        layer = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.5, 0.6],
                      [-0.1, 0.1],
                      [-0.2, 0.7]]),
            af.sigmoid)
        r = layer.feed_forward(np.array([[1, -1, 2]]))
        expected = np.array([af.sigmoid(0.2), af.sigmoid(1.9)])
        self.assertTrue(np.allclose(r, expected))

    def test_clone_equal(self):
        layer1 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.1, 0.2],
                      [-0.1, 0.1],
                      [-0.2, 0.1]]),
            af.sigmoid)

        layer2 = layer1.clone()
        self.assertEqual(layer1, layer2)

        # check the clone is not changing the original by
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

    def test_equal(self):
        np.random.seed(1)
        layer1 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.1, 0.2],
                      [-0.1, 0.1],
                      [-0.2, 0.1]]),
            af.sigmoid)
        layer2 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.1, 0.2],
                      [-0.1, 0.1],
                      [-0.2, 0.1]]),
            af.sigmoid)
        self.assertEqual(layer1, layer2)


class TestNeuralNet(TestCase):

    def test_clone_equal(self):
        net1 = nn.NeuralNet.createRandomNeuralNet(5, 6, 1, 10)
        net2 = net1.clone()
        self.assertEqual(net1, net2)
        net2.mutate(0.5)
        self.assertNotEqual(net1, net2)

    def test_feed_forward(self):
        input = np.array([[1, -1, 2]])
        layer1 = nn.NeuralLayer.createCustomLayer(
            3,
            np.array([[0.5, 0.6],
                      [-0.1, 0.1],
                      [-0.2, 0.7]]),
            lambda x: x)
        layer2 = nn.NeuralLayer.createCustomLayer(
            2,
            np.array([[0.1, 0.1],
                      [1, 1]]),
            lambda x: x)
        layer3 = nn.NeuralLayer.createCustomLayer(
            2,
            np.array([[0.1],
                      [0.2]]),
            lambda x: x - 0.1)
        layers = [layer1, layer2, layer3]
        net = nn.NeuralNet(layers)
        result = net.feed_forward(input)
        expected = np.array([[0.476]])
