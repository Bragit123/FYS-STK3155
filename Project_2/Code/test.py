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
key = random.PRNGKey(123)
# the number of datapoints
n = 100
x = 2*random.uniform(key,shape=(n,1))
x = jnp.sort(x)
y = 4+3*x+2*x**2+random.normal(key,shape=(n,1))


deg = 2
poly = PolynomialFeatures(degree=deg)
X = poly.fit_transform(x) # Find feature matrix
def CostOLS(beta):
    return jnp.sum((y-X @ beta)**2)

#beta = random.normal(key,shape=(deg+1,1))


n_iter=1000
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
gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter,algorithm="ADAM")
print("beta from GD, using Adam")
print(beta)

gd_class = Gradient_Descent(X,y,CostOLS,deg)
gd_class.set_tuning_parameters(eta=0.001)
beta = gd_class.GD(n_iter,momentum=0.001,algorithm="ADAM")
print("beta from GD with momentum, using Adam")
print(beta)


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
