import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

x = np.random.rand(100,1)[:,0]
y = 5*x**0.5-x**2+x**-(1.0/3.0)+.005*np.random.randn(100,1)[:,0]

fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel(r'$X_Values$')
ax.set_ylabel(r'$Y_Values$')
ax.scatter(x,y, alpha=0.7, lw=2,
            label='Generated Data')
plt.show()
