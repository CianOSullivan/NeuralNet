"""Shows the effect of bias on the sigmoid function
   From page 6 of the document
"""

import matplotlib.pylab as plt
import numpy as np

# Weights
w = 5.0
b1 = -8.0
b2 = 0.0
b3 = 8.0
# Labels
l1 = 'bias = -8.0'
l2 = 'bias = 0.0'
l3 = 'bias = 8.0'

# Returns evenly spaced values from -8 to 8
x = np.arange(-8, 8, 0.1)
# List of weights with their corresponding labels
weights = [(b1, l1), (b2, l2), (b3, l3)]

for bias, label in weights:
    f = 1 / (1 + np.exp(-x*w+bias)) # The sigmoid function
    plt.plot(x, f, label=label)

# Draw the plot
plt.xlabel('x')
plt.ylabel('func_weight(x)')
plt.legend() # Uses the label parameter from plt.plot to draw a legend
plt.show()
