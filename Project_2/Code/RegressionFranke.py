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
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score

seed = np.random.seed(200)
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

#Defining our input and target, and splitting the data
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
etas = np.logspace(-4,-1,4)
lmbds = np.logspace(-6,-1,6)
MSE_sigmoid = np.zeros((len(etas),len(lmbds)))
R2_sigmoid = np.zeros((len(etas),len(lmbds)))
MSE_RELU = np.zeros((len(etas),len(lmbds)))
R2_RELU = np.zeros((len(etas),len(lmbds)))
MSE_LRELU = np.zeros((len(etas),len(lmbds)))
R2_LRELU = np.zeros((len(etas),len(lmbds)))
#Training with our neural network code
for i in range(len(etas)):
    for j in range(len(lmbds)):
        scheduler = AdamMomentum(etas[i], rho, rho2, momentum = momentum)
        #sigmoid
        Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=200)
        scores = Neural.train(X_train, t_train, scheduler, batches=20, epochs=100, lam=lmbds[j], X_val = X_test, t_val = t_test)
        MSE_sigmoid[i,j] = scores["val_errors"][-1]
        output = Neural.predict(X_test)
        R2_sigmoid[i,j] = sklearn.metrics.r2_score(t_test, output)

        #RELU
        Neural = FFNN(dim, hidden_act=RELU, output_act=identity, cost_func=CostOLS, seed=200)
        scores = Neural.train(X_train, t_train, scheduler, batches=20, epochs=100, lam=lmbds[j], X_val = X_test, t_val = t_test)
        MSE_RELU[i,j] = scores["val_errors"][-1]
        output = Neural.predict(X_test)
        R2_RELU[i,j] = sklearn.metrics.r2_score(t_test, output)

        #LRELU
        Neural = FFNN(dim, hidden_act=LRELU, output_act=identity, cost_func=CostOLS, seed=200)
        scores = Neural.train(X_train, t_train, scheduler, batches=20, epochs=100, lam=lmbds[j], X_val = X_test, t_val = t_test)
        MSE_LRELU[i,j] = scores["val_errors"][-1]
        output = Neural.predict(X_test)
        R2_LRELU[i,j] = sklearn.metrics.r2_score(t_test, output)

heatmap(MSE_sigmoid, xticks=lmbds, yticks=etas, title="MSE test, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/MSE,Franke,sigmoid.pdf")
heatmap(R2_sigmoid, xticks=lmbds, yticks=etas, title="R2-score, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/R2,Franke,sigmoid.pdf")

heatmap(MSE_RELU, xticks=lmbds, yticks=etas, title="MSE test, RELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/MSE,Franke,RELU.pdf")
heatmap(R2_RELU, xticks=lmbds, yticks=etas, title="R2-score, RELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/R2,Franke,RELU.pdf")

heatmap(MSE_LRELU, xticks=lmbds, yticks=etas, title="MSE test, LRELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/MSE,Franke,LRELU.pdf")
heatmap(R2_LRELU, xticks=lmbds, yticks=etas, title="R2-score, LRELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/R2,Franke,LRELU.pdf")

#Scikit
# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 50
n_categories = 1
n_features = 2
epochs=100

eta_vals = np.logspace(-4, -1, 4)
lmbd_vals = np.logspace(-6, -1, 6)
# store models for later use
DNN_scikit_sigmoid = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
DNN_scikit_RELU = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        #sigmoid as activation function
        dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation="logistic", solver="adam",
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs,batch_size=5, momentum=0.01)
        dnn.fit(X_train, t_train)
        DNN_scikit_sigmoid[i][j] = dnn

        #RELU
        dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation="relu", solver="adam",
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs,batch_size=5, momentum=0.01)
        dnn.fit(X_train, t_train)
        DNN_scikit_RELU[i][j] = dnn


sns.set()
MSE_sigmoid = np.zeros((len(eta_vals), len(lmbd_vals)))
R2_sigmoid = np.zeros((len(eta_vals), len(lmbd_vals)))
MSE_RELU = np.zeros((len(eta_vals), len(lmbd_vals)))
R2_RELU = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        #sigmoid
        dnn = DNN_scikit_sigmoid[i][j]
        test_pred = dnn.predict(X_test)
        MSE_sigmoid[i,j] = sklearn.metrics.mean_squared_error(t_test,test_pred)
        R2_sigmoid[i,j] = sklearn.metrics.r2_score(t_test, test_pred)

        #RELU
        dnn = DNN_scikit_RELU[i][j]
        test_pred = dnn.predict(X_test)
        MSE_RELU[i,j] = sklearn.metrics.mean_squared_error(t_test,test_pred)
        R2_RELU[i,j] = sklearn.metrics.r2_score(t_test, test_pred)

#RELU
heatmap(MSE_RELU, xticks=lmbd_vals, yticks=eta_vals, title="MSE test with scikit, RELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/ScikitMSErelu.pdf")
heatmap(R2_RELU, xticks=lmbd_vals, yticks=eta_vals, title="R2-score with scikit, RELU", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/ScikitR2relu.pdf")
#sigmoid
heatmap(MSE_sigmoid, xticks=lmbd_vals, yticks=eta_vals, title="MSE test with scikit, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/ScikitMSEsigmoid.pdf")
heatmap(R2_sigmoid, xticks=lmbd_vals, yticks=eta_vals, title="R2-score with scikit, sigmoid", xlabel="$\lambda$", ylabel="$\eta$", filename="../Figures/ScikitR2sigmoid.pdf")

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
eta = 0.1
scheduler = AdamMomentum(eta, rho, rho2, momentum = 0.01)
dim = (2, 50, 1)

Neural = FFNN(dim, hidden_act=sigmoid, output_act=identity, cost_func=CostOLS, seed=200)

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

"""
err0 = scores["train_errors"][0]
err1 = scores["train_errors"][-1]
costf = CostCrossEntropy(target)
cost = costf(output)
print(f"cost = {cost}")
print(f"err0 = {err0} ; err1 = {err1}")
"""
