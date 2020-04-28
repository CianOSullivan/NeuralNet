"""
A neural network implementation for analysing the MNIST data set.
"""
from sklearn.datasets import load_digits              # Used to load the digits data
from sklearn.preprocessing import StandardScaler      # Used to scale the input data
from sklearn.model_selection import train_test_split  #
from sklearn.metrics import accuracy_score
import numpy as np                                    # This import takes a massive amount of time in python3 but not in python2
import matplotlib.pylab as plt
import time


def convert_out_to_vect(output):
    outVect = np.zeros((len(output), 10))
    for index in range(0, len(output)):
        outVect[index, output[index]] = 1
    return outVect


def f(x):
    """ Sigmoid function """
    return 1 / (1 + np.exp(-x))


def f_deriv(x):
    return f(x) * (1 - f(x))


def setup_init_weights(netStructure):
    """ Setup a random set of weights and biases for each layer of the network structure """
    weights = {}
    bias = {}

    for layer in range(1, len(netStructure)):
        weights[layer] = np.random.random_sample((netStructure[layer], netStructure[layer - 1]))
        # The trailing comma makes this a tuple
        bias[layer] = np.random.random_sample((netStructure[layer],))

    return weights, bias


def setup_delta_values(netStructure):
    """ Setup an empty set of detla weight and detla bias for each layer of the net structure """
    delta_wght = {}
    delta_bias = {}

    for layer in range(1, len(netStructure)):
        delta_wght[layer] = np.zeros((netStructure[layer], netStructure[layer - 1]))
        # The trailing comma makes this a tuple
        delta_bias[layer] = np.zeros((netStructure[layer],))

    return delta_wght, delta_bias


def feed_forward(x, weights, bias):
    h = {1: x}  # Set layer 1 to the input
    z = {}
    for layer in range(1, len(weights) + 1):
        # Setup the input array which the weights will be multiplied by for each layer
        if layer == 1:
            # If first layer, set the input to the x input vector
            node_in = x
        else:
            # If not the first layer, set the input to the output of previous layer
            node_in = h[layer]

        z[layer + 1] = weights[layer].dot(node_in) + bias[layer]
        h[layer + 1] = f(z[layer + 1])

    return h, z


def calc_output_delta(y, h_out, z_out):
    return -(y - h_out) * f_deriv(z_out)


def calc_hidden_delta(delta_plus1, w_l, z_l):
    return np.dot(np.transpose(w_l), delta_plus1) * f_deriv(z_l)


def train_network(netStructure, X, y, iter_num=4000, alpha=0.25):
    W, b = setup_init_weights(netStructure)
    cnt = 0     # The number of training iterations
    m = len(y)  # The length of output training vector
    avg_cost_func = []
    print('Starting gradient descent for {} iterations'.format(iter_num))
    while cnt < iter_num:
        # Update progress of training
        if cnt % 500 == 0:
            print('Iteration {} of {}'.format(cnt, iter_num))

        delta_wght, delta_bias = setup_delta_values(netStructure)
        avg_cost = 0

        for i in range(len(y)):
            delta = {}
            # perform the feed forward pass and return the stored h and z values, to be used in the
            # gradient descent step
            h, z = feed_forward(X[i, :], W, b)

            # loop from nl-1 to 1 backpropagating the errors
            for layer in range(len(netStructure), 0, -1):
                if layer == len(netStructure):
                    delta[layer] = calc_output_delta(y[i, :], h[layer], z[layer])
                    avg_cost += np.linalg.norm((y[i, :] - h[layer]))
                else:
                    if layer > 1:
                        delta[layer] = calc_hidden_delta(delta[layer + 1], W[layer], z[layer])
                    # deltaW^(l) = deltaW^(l) + delta^(l+1) * transpose(h^(l))
                    delta_wght[layer] += np.dot(delta[layer + 1][:, np.newaxis],
                                                np.transpose(h[layer][:, np.newaxis]))
                    # delta_b^(l) = delta_b^(l) + delta^(l+1)
                    delta_bias[layer] += delta[layer + 1]

        # Perform the gradient descent step for the weights in each layer
        for layer in range(len(netStructure) - 1, 0, -1):
            W[layer] += -alpha * (1.0 / m * delta_wght[layer])
            b[layer] += -alpha * (1.0 / m * delta_bias[layer])
        # complete the average cost calculation
        avg_cost = 1.0 / m * avg_cost
        avg_cost_func.append(avg_cost)
        cnt += 1

    return W, b, avg_cost_func


def predict_y(W, b, X, n_layers):
    m = X.shape[0]
    y = np.zeros((m,))
    for i in range(0, m):
        h, z = feed_forward(X[i, :], W, b)
        y[i] = np.argmax(h[n_layers])

    return y


def main():
    # Time the training of the model
    startTime = time.time()
    # Load the digits data set
    digits = load_digits()
    # Scale the digits data
    scaler = StandardScaler()
    inData = scaler.fit_transform(digits.data)  # The scaled digits data
    outData = digits.target

    # Initialise the input data and the desired output
    inTrain, inTest, outTrain, outTest = train_test_split(inData, outData, test_size=0.4)
    outTrainVect = convert_out_to_vect(outTrain)  # The vector which inTrain is compared against
    outTestVect = convert_out_to_vect(outTest)    # The vector which inTest is compared against
    print("Created training vector of {} items and test vector of {} items".format(outTrain.size,
                                                                                   outTest.size))
    print(outTrainVect[0:9])

    # Create the structure of the NN
    netStructure = [64, 30, 10]

    W, b, avg_cost_func = train_network(netStructure, inTrain, outTrainVect)
    print("Model setup and trained in %.1f seconds" % round(time.time() - startTime, 1))
    y_pred = predict_y(W, b, inTest, 3)
    print('Prediction accuracy is {}%\n'.format(accuracy_score(outTest, y_pred) * 100))

    while True:
        print("What would you like to do?: ")
        print("    (1) Visualise average cost function")
        print("    (2) Predict a number")
        print("    (3) Exit the program")

        answer = input("-> ")
        if answer == "1":
            # Plot the given number as an example
            plt.plot(avg_cost_func)
            plt.ylabel('Average J')
            plt.xlabel('Iteration number')
            plt.show()
        elif answer == "2":
            print(inData)
            print("")
            print(digits.data)
            print("Predicting a number")
        elif answer == "3":
            exit()
        else:
            print("Please enter a valid number")


try:
    main()
except KeyboardInterrupt:
    print("\nProgram Exit")
    exit()
