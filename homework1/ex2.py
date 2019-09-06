#Machine Learning HW1 exercise 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

xstart = np.random.rand(100,1)[:,0]
#creates [100,1] array and puts it in column form
x = np.sort(xstart, axis = None)
#sorts data points from small to big

y = 5*x**0.5-x**2+x**(1.0/3.0)+.5*np.random.randn(100,1)[:,0]
#creates date set to work with, we will be trying to make a polynomial fit
#to these points
print(x,'\n\n',y,'\n\n')

# create poynimial fit design matrix
xmat = np.zeros((len(x),5))
xmat[:,0]=1
xmat[:,1]=x
xmat[:,2]=x**2
xmat[:,3]=x**3
xmat[:,4]=x**4

print(xmat)

#actually solve using our design matrix and derived beta vector
beta = np.linalg.inv(xmat.T.dot(xmat)).dot(xmat.T).dot(y)
ytilde = xmat @ beta


# Generate a plot comparing the experimental with the fitted values values.
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel(r'$X_Values$')
ax.set_ylabel(r'$Y_Values$')
ax.scatter(x,y, alpha=0.7, lw=2,
            label='Generated Data')
ax.plot(x,ytilde, alpha=0.7, lw=2, c='m',
            label='Fit')
ax.legend()
plt.show()
