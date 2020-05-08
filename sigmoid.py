"""Sigmoid function"""

import matplotlib.pylab as plt
import numpy as np


x = np.arange(-8, 8, 0.1) # Step from -8 to 8 in 0.1 intervals
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
