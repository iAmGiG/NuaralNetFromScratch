import os
import math
from absl import app
from absl import flags
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

flags.DEFINE_integer("Random_seed", 0, "The seed given to np.random.seed, default is 0")

FLAGS = flags.FLAGS


def rec_linear(inputs, output):
    """
    rectify linear function
    :param inputs: the tensor input
    :param output: the list for holding the output
    :return: the output will have either 0 or a value larger than 0 in the list.
    """
    for index in inputs:
        output.append(max(0, index))
    return output


def soft_max(layer_outputs):
    """
    softmax uses Euler's number to keep the meaning of negative values without using the negative value itself
    :param layer_outputs: the output
    :return: exponential values
    """
    exp_values = []
    for output in layer_outputs:
        exp_values.append(math.e ** output)
    return exp_values


class ActivationSoftMax:
    def forward(self, inputs):
        """
        :param inputs: the model outputs
        :return:
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probablities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probablities


class ActivationReLu:
    def forward(self, inputs):
        """
        :param inputs:
        :return:
        """
        self.output = np.maximum(0, inputs)


class LayerDense:
    def __init__(self, n_input, n_neurons):
        """
        :param n_input:
        :param n_neurons:
        :return:
        """
        self.weights = 0.1 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        :param inputs:
        :return:
        """
        self.output = np.dot(inputs, self.weights) + self.biases


def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}")

    # Random seed override: np.random.seed(FLAGS.Random_seed)
    '''
    layer_output = [[4.8, 1.21, 2.385],
                    [8.9, 1.81, 0.2],
                    [1.41, 1.051, 0.026]]
    exp_values = np.exp(layer_output)

    norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    print(norm_values)
    '''
    X, y = spiral_data(samples=100, classes=3)
    dense1 = LayerDense(2, 3)
    activation1 = ActivationReLu()

    dense2 = LayerDense(3, 3)
    activation2 = ActivationSoftMax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    
    print(activation2.output[:5])

    os._exit(0)


if __name__ == '__main__':
    nnfs.init()
    app.run(main)
