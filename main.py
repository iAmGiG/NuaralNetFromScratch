from abc import ABCMeta, abstractmethod
import os
import numpy as np
from absl import app
from absl import flags
import nnfs
from nnfs.datasets import spiral_data

flags.DEFINE_integer("Random_seed", 0, "The seed given to np.random.seed, default is 0")

FLAGS = flags.FLAGS


def create_data(points, classes):
    X = np.zeros((points * classes), 2)
    y = np.zeros(points * classes, dtype='units8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) * np.random.randn(points) * .2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


def current_output(inputs, weights, biases):
    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

    return layer_outputs


def rec_linear(inputs, output):
    for index in inputs:
        output.append(max(0, index))
    return output


def quick_dot_prod(inputs, weights, bias):
    output = np.dot(inputs, np.array(weights).T) + bias
    return output


class ActivationReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class LayerDense:
    def __init__(self, n_input, n_neurons):
        '''
        :param n_input:
        :param n_neurons:
        :return:
        '''
        self.weights = 0.1 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        '''
        :param inputs:
        :return:
        '''
        self.output = np.dot(inputs, self.weights) + self.biases


def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}")

    np.random.seed(FLAGS.Random_seed)
    '''
    X = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

    layer1 = LayerDense(4, 5)
    layer2 = LayerDense(5, 4)

    layer1.forward(X)
    layer2.forward(layer1.output)
    print(layer2.output)
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
    '''
    X, y = spiral_data(100, 2)
    layer1 = LayerDense(2, 5)
    activation1 = ActivationReLu()
    layer1.forward(X)
    activation1.forward(layer1.output)
    print(activation1.output)
    os._exit(0)


if __name__ == '__main__':
    nnfs.init()
    app.run(main)
