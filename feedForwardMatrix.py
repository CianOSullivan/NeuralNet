""" 
Run a feed forward pass using matrices with a given input
"""

import numpy as np # This import takes a massive amount of time in python3 but not in python2

def f(x):
    return 1 / (1 + np.exp(-x))

""" I think the shape attribute of numpy arrays gives the number of rows and columns of the array
    Therefore position 0 of the shape attribute would be the rows and position 1 would be the
    columns
"""


def matrix_feed_forward_calc(n_layers, x, weights, bias):
    for layer in range(0, n_layers-1):
        #Setup the input array which the weights will be multiplied by for each layer
        if layer == 0:
            # Set the input array to the x input vector in first layer
            node_in = x
        else:
            # Set the input to the next layer to the output of previous layer
            node_in = h

        z = weights[layer].dot(node_in) + bias[layer]
        h = f(z)

    return h

# Weights
w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
# Make a matrix with three rows set to 0
w2 = np.array([[0.5, 0.5, 0.5]])
weights = [w1, w2]

# Biases
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])
bias = [b1, b2]

#a dummy x input vector
x = [1.5, 2.0, 3.0]
print(matrix_feed_forward_calc(3, x, weights, bias))
