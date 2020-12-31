import sys
import os
import numpy as np
import matplotlib
from absl import app


def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}")

    inputs = [1, 2, 3, 2.5]
    weights0 = [0.2, 0.8, -0.5, 1.0]
    weights1 = [0.5, -0.91, 0.26, -0.5]
    weights2 = [-0.26, -0.27, .17, 0.87]
    bias0 = 2
    bias1 = 3
    bias2 = 0.5

    output = [(inputs[0] * weights0[0]) +
              (inputs[1] * weights0[1]) +
              (inputs[2] * weights0[2]) +
              (inputs[3] * weights0[3]) + bias0,
              (inputs[0] * weights1[0]) +
              (inputs[1] * weights1[1]) +
              (inputs[2] * weights1[2]) +
              (inputs[3] * weights1[3]) + bias1,
              (inputs[0] * weights2[0]) +
              (inputs[1] * weights2[1]) +
              (inputs[2] * weights2[2]) +
              (inputs[3] * weights2[3]) + bias2
              ]

    print(output)

    os._exit(0)


if __name__ == '__main__':
    app.run(main)
