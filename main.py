import sys
import os
import numpy as np
import matplotlib
from absl import app

def main(argv):
    if len(argv) > 2:
        raise app.UsageError("Expected one command-line argument(s), "
                             f"got: {argv}")

    inputs = [8.8, 99.44, 17.1]
    weights = [1, 2, 3]
    bias = 3

    output = ((inputs[0] * weights[0]) + (inputs[1] * weights[1]) + (inputs[2] * weights[2]) + 3)

    print(output)

    os._exit(0)


if __name__ == '__main__':
    app.run(main)
