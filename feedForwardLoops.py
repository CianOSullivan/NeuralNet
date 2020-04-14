""" 
Run a feed forward pass using python loops over each nodes output with a given input
"""

import matplotlib.pylab as plt
import numpy as np

def f(x):
    return 1 / (1 + np.exp(-x))

""" I think the shape attribute of numpy arrays gives the number of rows and columns of the array
    Therefore position 0 of the shape attribute would be the rows and position 1 would be the
    columns
"""


def simple_looped_nn_calc(n_layers, x, weights, bias):
    for layer in range(0, n_layers-1):
        #Setup the input array which the weights will be multiplied by for each layer
        if layer == 0:
            # Set the input array to the x input vector in first layer
            node_in = x
        else:
            # Set the input to the next layer to the output of previous layer
            node_in = h
            
        #Setup the output array for the nodes in layer l + 1
        h = np.zeros((weights[layer].shape[0],))
        
        #loop through the rows of the weight array
        for i in range(weights[layer].shape[0]):
            #setup the sum inside the activation function
            f_sum = 0
            #loop through the columns of the weight array
            for j in range(weights[layer].shape[1]):
                f_sum += weights[layer][i][j] * node_in[j]
            #add the bias
            f_sum += bias[layer][i]
            #finally use the activation function to calculate the
            #i-th output i.e. h1, h2, h3
            h[i] = f(f_sum)
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
print(simple_looped_nn_calc(3, x, weights, bias))
