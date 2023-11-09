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
from scheduler import *
from funcs import *
from copy import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sklearn
from plotting import *
"""
def f(x):
    return 4.*x**2 + 3.*x + 6.
x = np.linspace(0,1,10)
x = x.reshape(len(x),1)
target = f(x)

rho = 0.9
rho2 = 0.999
eta = 0.01
scheduler = Adam(eta, rho, rho2)
dim = (1, 7, 7, 7, 1)

# Neural = FFNN(dim, act_func=sigmoid, cost_func=CostCrossEntropy, seed=100)
Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=100)

#output = Neural.predict(x)
#print("Before backpropagation")
#print(output)

scores = Neural.train(x, target, scheduler, epochs=1000, lam=0)

output = Neural.predict(x)
print("After backpropagation")
print(output,target)
plt.plot(x,target)
plt.plot(x,output)
plt.show()

epochs = np.arange(len(scores["train_errors"]))
plt.plot(epochs, scores["train_errors"])
plt.show()
"""

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

#Calculating MSE, R2, Sigmoid
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
x_shape = x.shape
y_shape = y.shape
x = x.flatten()
y = y.flatten()
X = np.array([x.flatten(),y.flatten()]).T
target = FrankeFunction(x.flatten(),y.flatten())
target += 0.1*np.random.normal(0, 1, target.shape) #Adding noise
target = target.reshape((len(target),1))
X_train, X_test, t_train, t_test = train_test_split(X, target, test_size=0.2)
#Adam parameters
rho = 0.9
rho2 = 0.999
momentum = 0.01
dim = (2, 50, 1)
etas = np.logspace(-3,0,4)
lmbds = np.logspace(-4,0,5)
MSE = np.zeros((len(etas),len(lmbds)))
R2 = np.zeros((len(etas),len(lmbds)))
for i in range(len(etas)):
    for j in range(len(lmbds)):
        scheduler = AdamMomentum(etas[i], rho, rho2, momentum = momentum)
        Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=100)
        #output = Neural.predict(X_train) #before backprop
        scores = Neural.train(X_train, t_train, scheduler, batches=20, epochs=100, lam=lmbds[j], X_val = X_test, t_val = t_test)
        MSE[i,j] = scores["val_errors"][-1]
        output = Neural.predict(X_test)
        R2[i,j] = sklearn.metrics.r2_score(t_test, output)

heatmap(MSE, xticks=lmbds, yticks=etas, title="MSE test, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/MSE,Franke,sigmoid.pdf")
heatmap(R2, xticks=lmbds, yticks=etas, title="R2-score, sigmoid", xlabel="$\lamdba$", ylabel="$\eta$", filename="../Figures/R2,Franke,sigmoid.pdf")

#Calculating MSE, R2, RELU
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
x_shape = x.shape
y_shape = y.shape
x = x.flatten()
y = y.flatten()
X = np.array([x.flatten(),y.flatten()]).T
target = FrankeFunction(x.flatten(),y.flatten())
target += 0.1*np.random.normal(0, 1, target.shape) #Adding noise
target = target.reshape((len(target),1))
X_train, X_test, t_train, t_test = train_test_split(X, target, test_size=0.2)
#Adam parameters
rho = 0.9
rho2 = 0.999
momentum = 0.01
dim = (2, 50, 1)
etas = np.logspace(-3,0,4)
lmbds = np.logspace(-4,0,5)
MSE = np.zeros((len(etas),len(lmbds)))
R2 = np.zeros((len(etas),len(lmbds)))
for i in range(len(etas)):
    for j in range(len(lmbds)):
        scheduler = AdamMomentum(etas[i], rho, rho2, momentum = momentum)
        Neural = FFNN(dim, hidden_act=RELU, output_act=identity, cost_func=CostOLS, seed=100)
        #output = Neural.predict(X_train) #before backprop
        scores = Neural.train(X_train, t_train, scheduler, batches=20, epochs=100, lam=lmbds[j], X_val = X_test, t_val = t_test)
        MSE[i,j] = scores["val_errors"][-1]
        output = Neural.predict(X_test)
        R2[i,j] = sklearn.metrics.r2_score(t_test, output)

heatmap(MSE, xticks=lmbds, yticks=etas, title="MSE test, RELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/MSE,Franke,RELU.pdf")
heatmap(R2, xticks=lmbds, yticks=etas, title="R2-score, RELU", xlabel="$\lamba$", ylabel="$\eta$", filename="../Figures/R2,Franke,RELU.pdf")



#Calculating MSE, R2, LRELU
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
x_shape = x.shape
y_shape = y.shape
x = x.flatten()
y = y.flatten()
X = np.array([x.flatten(),y.flatten()]).T
target = FrankeFunction(x.flatten(),y.flatten())
target += 0.1*np.random.normal(0, 1, target.shape) #Adding noise
target = target.reshape((len(target),1))
X_train, X_test, t_train, t_test = train_test_split(X, target, test_size=0.2)
#Adam parameters
rho = 0.9
rho2 = 0.999
momentum = 0.01
dim = (2, 50, 1)
etas = np.logspace(-3,0,4)
lmbds = np.logspace(-4,0,5)
MSE = np.zeros((len(etas),len(lmbds)))
R2 = np.zeros((len(etas),len(lmbds)))
for i in range(len(etas)):
    for j in range(len(lmbds)):
        scheduler = AdamMomentum(etas[i], rho, rho2, momentum = momentum)
        Neural = FFNN(dim, hidden_act=LRELU, output_act=identity, cost_func=CostOLS, seed=100)
        #output = Neural.predict(X_train) #before backprop
        scores = Neural.train(X_train, t_train, scheduler, batches=20, epochs=100, lam=lmbds[j], X_val = X_test, t_val = t_test)
        MSE[i,j] = scores["val_errors"][-1]
        output = Neural.predict(X_test)
        R2[i,j] = sklearn.metrics.r2_score(t_test, output)

heatmap(MSE, xticks=lmbds, yticks=etas, title="MSE test, LRELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/MSE,Franke,LRELU.pdf")
heatmap(R2, xticks=lmbds, yticks=etas, title="R2-score, LRELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/R2,Franke,LRELU.pdf")


#Visualising the fit for parameters we found to give a small MSE
x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
x_shape = x.shape
y_shape = y.shape
x = x.flatten()
y = y.flatten()
X = np.array([x.flatten(),y.flatten()]).T
target = FrankeFunction(x.flatten(),y.flatten())
target += 0.1*np.random.normal(0, 1, target.shape) #Adding noise
target_shape = target.shape
target = target.reshape((len(target),1))

rho = 0.9
rho2 = 0.999
eta=0.1
scheduler = AdamMomentum(eta, rho, rho2, momentum = 0.01)
dim = (2, 50, 1)

Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=100)

#output = Neural.predict(X_train) #before backprop

scores = Neural.train(X, target, scheduler, batches=20, epochs=100, lam=1e-5)


output = Neural.predict(X) #After backprop

output = output.reshape(x_shape)
x = x.reshape(x_shape)
y = y.reshape(y_shape)
#print(output.shape)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(x, y, z, label="Franke Function",cmap=cm.coolwarm, linewidth=0, antialiased=False)
turf = ax.plot_surface(x, y, output,label="Fit from FFNN",cmap=cm.viridis, linewidth=0, antialiased=False)
ax.set_zlim(-0.10, 1.40)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Fit for Franke function")
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.legend()
plt.savefig("../Figures/Frankefitvisualisation.pdf")
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
