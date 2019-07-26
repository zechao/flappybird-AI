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
