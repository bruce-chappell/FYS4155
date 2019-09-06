import numpy as np
from franke import FrankeFunction
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.gca(projection='3d')

nx,ny = (25,25)
x = np.linspace(0,1,nx)
y = np.linspace(0,1, ny)
x, y = np.meshgrid(x,y)

z=FrankeFunction(x,y)+0.2*np.random.randn(25,25)

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

#print(z)
