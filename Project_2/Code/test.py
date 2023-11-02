import numpy as np
from GradientDescentSolver import Gradient_Descent
from NeuralNetwork import Node, Layer

from random import random, seed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from GradientDescentSolver import *
import jax.numpy as jnp
from jax import grad, random, jit

key = random.PRNGKey(456)
class NeuralNetwork:
    def __init__(self, X, y, cost_func, eta = 0.01,lmbd=0.01, n_hidden_neurons = 2, n_categories = 1, n_features = 2, gradient_func=None):
        self.X = X
        self.y = y
        self.eta = eta
        self.lmbd = lmbd
        self.cost_func = cost_func
        #self.deg = deg
        self.beta = None
        if gradient_func == None:
            self.gradient_func = grad(cost_func(y))
        else:
            self.gradient_func = gradient_func

        self.tuning_algorithm = Tuning_algorithm() # Index of 'active' tuning algorithm (chosen from tuning_algorithms)
        # Defining the neural network
        self.n_inputs, self.n_features = X.shape
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories  #Only one output node
        self.n_features = n_features

        # weights and bias in the hidden layer
        self.hidden_weights = random.normal(key,shape=(n_features,n_hidden_neurons))
        self.hidden_bias = jnp.zeros(n_hidden_neurons) + 0.01

        # weights and bias in the output layer
        self.output_weights = random.normal(key,shape=(n_hidden_neurons,n_categories))
        self.output_bias = jnp.zeros(n_categories) + 0.01

    def sigmoid(self,x):
        return 1/(1 + jnp.exp(-x))

    def feed_forward(self,X):
        # weighted sum of inputs to the hidden layer
        z_h = jnp.matmul(X, self.hidden_weights) + self.hidden_bias
        # activation in the hidden layer
        a_h = self.sigmoid(z_h)

        # weighted sum of inputs to the output layer
        z_o = jnp.matmul(a_h, self.output_weights) + self.output_bias
        # softmax output
        # axis 0 holds each input and axis 1 the probabilities of each category
        exp_term = jnp.exp(z_o)
        probabilities = exp_term / jnp.sum(exp_term, axis=1, keepdims=True)

        return a_h, probabilities

    # we obtain a prediction by taking the class with the highest likelihood
    def predict(self,X):
        probabilities = self.feed_forward(self.X)[1]
        return jnp.argmax(probabilities, axis=1)

    def backpropagate(self,n_iter):
        for i in range(n_iter):
            a_h, probabilities = self.feed_forward(X)
            target = self.y.reshape(-1,1)
            error_output = probabilities - target #grad_func(probabilities)
            error_hidden = jnp.matmul(error_output, self.output_weights.T) * a_h * (1 - a_h)

            # gradients for the output layer
            output_weights_gradient = jnp.matmul(a_h.T, error_output)
            output_bias_gradient = jnp.sum(error_output, axis=0)

            # gradient for the hidden layer
            hidden_weights_gradient = jnp.matmul(X.T, error_hidden)
            hidden_bias_gradient = jnp.sum(error_hidden, axis=0)

            #Regularization terms gradients
            output_weights_gradient += self.lmbd * self.output_weights
            hidden_weights_gradient += self.lmbd * self.hidden_weights

            # weights and bias in the hidden layer
            self.hidden_weights -= self.eta*hidden_weights_gradient
            self.hidden_bias -= self.eta*hidden_bias_gradient

            # weights and bias in the output layer
            self.output_weights -= self.eta*output_weights_gradient
            self.output_bias -= self.eta*output_bias_gradient
        a_h, prob_last = self.feed_forward(self.X)
        return prob_last

# Design matrix
X = jnp.array([ [0.0, 0.0], [0.0, 1.0], [1.0, 0.0],[1.0, 1.0]])

# The XOR gate
yXOR = jnp.array( [ 0.0, 1.0 ,1.0, 0.0])
# The OR gate
yOR = jnp.array( [ 0.0, 1.0 ,1.0, 1.0])
# The AND gate
yAND = jnp.array( [ 0.0, 0.0 ,0.0, 1.0])

def CostCrossEntropy(target):

    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func

n_iter = 100
neural_1 = NeuralNetwork(X,yXOR,CostCrossEntropy)
probabilities = neural_1.backpropagate(n_iter)
print("Probs, XOR")
print(probabilities)


key = random.PRNGKey(123)
# the number of datapoints
n = 100
x = 2*random.uniform(key,shape=(n,1))
x = jnp.sort(x)
y = 4+3*x+2*x**2#+random.normal(key,shape=(n,1))


deg = 2
poly = PolynomialFeatures(degree=deg)
X = poly.fit_transform(x) # Find feature matrix
def CostOLS(beta):
    return jnp.sum((y-X @ beta)**2)

#beta = random.normal(key,shape=(deg+1,1))


n_iter=100
XT_X = X.T @ X
beta_linreg = jnp.linalg.inv(X.T @ X) @ (X.T @ y)
print("beta from OLS")
print(beta_linreg)

"""
gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter)
print("beta from GD")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter,momentum=0.001)
print("beta from GD with momentum")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter,algorithm="RMSPROP")
print("beta from GD, using RMS-prop")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter,momentum=0.001,algorithm="RMSPROP")
print("beta from GD with momentum, using RMS-prop")
print(beta)
"""
def CostOLS2(beta,X,y):
    return jnp.sum((y - X @ beta)**2)
"""
gd_class = Gradient_Descent(X,y,CostOLS2,deg)
gd_class.set_tuning_parameters(eta=10**(-5))
beta = gd_class.SGD(n_iter,n_epochs=50, batch_size=5)
print("beta from SGD")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS2,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.SGD(n_iter,n_epochs=50, batch_size=5,momentum=0.001)
print("beta from SGD with momentum")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter,algorithm="ADAGRAD")
print("beta from GD, using Adagrad")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter,momentum=0.001,algorithm="ADAGRAD")
print("beta from GD with momentum, using Adagrad")
print(beta)
"""
"""
gd_class = Gradient_Descent(X,y,CostOLS2,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.SGD(n_iter,n_epochs=50,batch_size=5,algorithm="ADAM")
print("beta from SGD, using Adam")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS2,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.SGD(n_iter,n_epochs=50,batch_size=5,momentum=0.001,algorithm="ADAM")
print("beta from SGD with momentum, using Adam")
print(beta)

"""
"""
This is printed:
beta from OLS
[[2.0863998]
 [5.227591 ]
 [1.7393135]]
beta from GD
[[2.086448 ]
 [5.227451 ]
 [1.7393789]]
beta from GD with momentum
[[2.0863922]
 [5.2276073]
 [1.739303 ]]
beta from GD, using RMS-prop
[[0.47813433]
 [3.9825766 ]
 [0.03088689]]
beta from GD with momentum, using RMS-prop
[[2.0864348]
 [5.2274885]
 [1.7393607]]

"""
"""
deg = 2

def CostRidge(beta,X,y):
    return jnp.sum((y - X @ beta)**2)#+lmb*jnp.linalg.norm(beta)**2

lmbs = jnp.array([0.0001,0.001,0.01,0.1,1])
etas = jnp.array([0.1,0.01,0.001,0.0001,0.00001])

for lmb in lmbs:
    for eta in etas:
        gd_class = Gradient_Descent(X,y,CostRidge,deg)
        gd_class.set_tuning_parameters(eta=eta)
        beta = gd_class.SGD(n_iter,n_epochs=50,batch_size=5)
        print(f"lambda={lmb}, eta={eta}")
        print(beta)

        gd_class = Gradient_Descent(X,y,CostRidge,deg)
        gd_class.set_tuning_parameters(eta=eta)
        beta = gd_class.SGD(n_iter,n_epochs=50,batch_size=5,momentum=0.001)
        print(f"Mom, lambda={lmb}, eta={eta}")
        print(beta)
"""
