import sys
import os
import numpy as np
import matplotlib
from absl import app


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

def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}")

    inputs = [[1, 2, 3, 2.5],
				[2.0, 5.0, -1.0, 2.0],
				[-1.5, 2.7, 3.3, -0.8]]

    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, .17, 0.87]]
			   
    biases = [2, 3, 0.5]

	output = quick_dot_prod(inputs, weights, biases)

    layer_outputs = current_output(inputs, weights, biases)

    print(layer_outputs)

    os._exit(0)


if __name__ == '__main__':
    app.run(main)
