import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from random import random, seed
import jax.numpy as jnp
import pandas as pd
import seaborn as sns
from NN import FFNN
from scheduler import Constant, Adam
from funcs import CostCrossEntropy, sigmoid, CostLogReg, CostOLS
from copy import copy


def f(x):
    return 4.*x**2 + 3.*x + 6.
x = np.linspace(0,1,4)
x = x.reshape(len(x),1)
target = f(x)

rho = 0.9
rho2 = 0.999
eta = 0.01
scheduler = Adam(eta, rho, rho2)
dim = (1, 5, 1)

# Neural = FFNN(dim, act_func=sigmoid, cost_func=CostCrossEntropy, seed=100)
Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=100)

#output = Neural.predict(x)
#print("Before backpropagation")
#print(output)

scores = Neural.train(x, target, scheduler, epochs=100)

output = Neural.predict(x)
print("After backpropagation")
print(output,target)
plt.plot(x,target)
plt.plot(x,output)
plt.show()

## Making the Franke function. This part is largely copied from the projection description
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Generate data.
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)

def FrankeFunction(x, y):
    """ Calculates the Franke function at a point (x,y) """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

z = FrankeFunction(x, y) # Calculate the Franke function for our dataset.

# Plot the surface.
surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("../Figures/franke_function.pdf")

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
X = np.array([x,y]).T
x, y = np.meshgrid(x,y)
target = FrankeFunction(x,y)

rho = 0.9
rho2 = 0.999
eta = 0.01
scheduler = Adam(eta, rho, rho2)
dim = (2, 5, 1)

# Neural = FFNN(dim, act_func=sigmoid, cost_func=CostCrossEntropy, seed=100)
Neural = FFNN(dim, act_func=sigmoid, cost_func=CostOLS, seed=100)

output = Neural.predict(X)
print("Before backpropagation")
print(output)

scores = Neural.train(X, target, scheduler, epochs=1000)

output = Neural.predict(X)
print("After backpropagation")
print(output-target)


surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
turf = ax.plot_surface(x,y,output,cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
#plt.savefig("../Figures/franke_function.pdf")
# print("Scores")
# print(scores)
epochs = np.arange(len(scores["train_errors"]))
plt.plot(epochs, scores["train_errors"])
plt.show()

err0 = scores["train_errors"][0]
err1 = scores["train_errors"][-1]
costf = CostCrossEntropy(target)
cost = costf(output)
print(f"cost = {cost}")
print(f"err0 = {err0} ; err1 = {err1}")
