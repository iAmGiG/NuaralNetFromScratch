# %%
inputs = [1, 2, 3, 4, 5]
weight = [1, .02, .3, .4, -.5]
weight2 = [.01, .02, -.32, -.04, -.9]
weight3 = [.9, -.26, -.27, .17, .87]
bias = 2
bias2 = 3
bias3 = 0.5
output = [inputs[0] * weight[0] +
          inputs[1] * weight[1] +
          inputs[2] * weight[2] +
          inputs[3] * weight[3] +
          inputs[4] * weight[4] + bias,
          inputs[0] * weight2[0] +
          inputs[1] * weight2[1] +
          inputs[2] * weight2[2] +
          inputs[3] * weight2[3] +
          inputs[4] * weight2[4] + bias2,
          inputs[0] * weight3[0] +
          inputs[1] * weight3[1] +
          inputs[2] * weight3[2] +
          inputs[3] * weight3[3] +
          inputs[4] * weight3[4] + bias3]
print(output)
# %%
inputs = [1, 2, 3, 2.5]
weights = [[.2, .8, -.5, 1],
           [.5, -.91, .26, -.5],
           [-.26, -.27, .17, .87]]
biases = [2, 3, 0.5]
layer_outputs = []  # Output of current layer

for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # Output of given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)

# %%
import numpy as np
# dot_product = 
# %%
a, b = 0, 1
for i in range(2):
  print(a)
  a, b = b, a + b
# %%
