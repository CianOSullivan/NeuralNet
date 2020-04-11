"""Shows the effect of weighting on the sigmoid function
   From page 6 of the document
"""

import matplotlib.pylab as plt
import numpy as np

# Weights
w1 = 0.5
w2 = 1.0
w3 = 2.0
# Labels
l1 = 'weight = 0.5'
l2 = 'weight = 1.0'
l3 = 'weight = 2.0'

# Returns evenly spaced values from -8 to 8
x = np.arange(-8, 8, 0.1)
# List of weights with their corresponding labels
weights = [(w1, l1), (w2, l2), (w3, l3)]

for weight, label in weights:
    f = 1 / (1 + np.exp(-x*weight)) # The sigmoid function
    plt.plot(x, f, label=label)

# Draw the plot
plt.xlabel('x')
plt.ylabel('func_weight(x)')
plt.legend() # Uses the label parameter from plt.plot to draw a legend
plt.show()
