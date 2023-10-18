import numpy as np
from GradientDescentSolver import Gradient_Descent
from NeuralNetwork import Node, Layer

from random import random, seed
import numpy as np
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

beta = random.normal(key,shape=(deg+1,1))


n_iter=1000
XT_X = X.T @ X
beta_linreg = jnp.linalg.inv(X.T @ X) @ (X.T @ y)
print("beta from OLS")
print(beta_linreg)

gd_class = Gradient_Descent(X,y,CostOLS,beta)
beta = gd_class.GD(n_iter)
print("beta from GD")
print(beta)
