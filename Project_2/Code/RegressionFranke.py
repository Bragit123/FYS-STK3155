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


x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
x_shape = x.shape
y_shape = y.shape
x = x.flatten()
y = y.flatten()
X = np.array([x.flatten(),y.flatten()]).T
target = FrankeFunction(x.flatten(),y.flatten())
X_train, X_test, t_train, t_test = train_test_split(X, target, test_size=0.2)
#Adam parameters
#Calculating MSE
rho = 0.9
rho2 = 0.999
momentum = 0.01
dim = (2, 50, 1)
etas = np.logspace(-4,0,5)
lmbds = np.logspace(-4,0,5)
MSE = np.zeros((len(etas),len(lmbds)))
for i in range(len(etas)):
    for j in range(len(lmbds)):
        scheduler = AdamMomentum(eta[i], rho, rho2, momentum = momentum)
        Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=100)
        #output = Neural.predict(X_train) #before backprop
        scores = Neural.train(X_train, t_train, scheduler, batches=20, epochs=100, lam=lmbds[i], X_test, t_test)
        MSE[i,j] = scores["val_errors"]

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()


x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
x_shape = x.shape
y_shape = y.shape
x = x.flatten()
y = y.flatten()
X = np.array([x.flatten(),y.flatten()]).T
target = FrankeFunction(x.flatten(),y.flatten())
target_shape = target.shape
target = target.reshape((len(target),1))

rho = 0.9
rho2 = 0.999
eta=0.1
scheduler = AdamMomentum(eta, rho, rho2, momentum = 0.01)
dim = (2, 50, 1)

Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=100)

#output = Neural.predict(X_train) #before backprop

scores = Neural.train(X, t, scheduler, batches=20, epochs=100, lam=1e-5)


output = Neural.predict(X) #After backprop

output = output.reshape(x_shape)
#print(output.shape)

surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
turf = ax.plot_surface(x, y, output,cmap=cm.coolwarm, linewidth=0, antialiased=False)
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


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns

# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 2
n_features = 2

eta_vals = np.logspace(-5, 1, 7)
lmbd_vals = np.logspace(-5, 1, 7)
# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
epochs = 100

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPClassifier(hidden_layer_sizes=(n_hidden_neurons), activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn.fit(X, yXOR)
        DNN_scikit[i][j] = dnn
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on data set: ", dnn.score(X, yXOR))
        print()

sns.set()
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        test_pred = dnn.predict(X)
        test_accuracy[i][j] = accuracy_score(yXOR, test_pred)

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test Accuracy")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.show()
