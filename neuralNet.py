"""
A neural network implementation for analysing the MNIST data set.
"""
from sklearn.datasets import load_digits             # Used to load the digits data
from sklearn.preprocessing import StandardScaler     # Used to scale the input data
from sklearn.model_selection import train_test_split #
import numpy as np                                   # This import takes a massive amount of time in python3 but not in python2
import matplotlib.pylab as plt


def convert_out_to_vect(output):
    outVect = np.zeros((len(output), 10))
    for index in range(0, len(output)):
        outVect[index, output[index]] = 1
    return outVect


def f(x):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-x))

def f_deriv(x):
    return f(x) * (1-f(x))


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


def main():
    # Load the digits data set
    digits = load_digits()
    # Scale the digits data
    scaler = StandardScaler()
    inData = scaler.fit_transform(digits.data) # The scaled digits data
    outData = digits.target
    
    # Initialise the input data and the desired output
    inTrain, inTest, outTrain, outTest = train_test_split(inData, outData, test_size=0.4)
    outTrainVect = convert_out_to_vect(outTrain) # The vector which inTrain is compared against
    outTestVect = convert_out_to_vect(outTest) # The vector which inTest is compared against
    print("Created training vector of size: ", outTrain.size)
    print("Created test vector of size: ", outTest.size)

    print(outTrainVect[0:9])
    #plt.gray()
    #plt.matshow(digits.images[1])
    #plt.show()


main()
