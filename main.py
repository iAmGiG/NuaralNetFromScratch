import sys
import os
import numpy as np
import matplotlib
from absl import app
from absl import flags

flags.DEFINE_integer("Random_seed", 0, "The seed given to np.random.seed, default is 0")

FLAGS = flags.FLAGS


def current_output(inputs, weights, biases):
    layer_outputs = []

    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight
        neuron_output += neuron_bias
        layer_outputs.append(neuron_output)

    return layer_outputs


def quick_dot_prod(inputs, weights, bias):
    output = np.dot(inputs, np.array(weights).T) + bias
    return output


class Layer_Dense:
    def __inti__(self, n_input, n_neurons):
        self.weights = 0.1 * np.random.randn(n_input, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        pass


def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}")
    np.random.seed(FLAGS.Random_seed)
    X = [[1, 2, 3, 2.5],
         [2.0, 5.0, -1.0, 2.0],
         [-1.5, 2.7, 3.3, -0.8]]

    os._exit(0)


if __name__ == '__main__':
    app.run(main)
