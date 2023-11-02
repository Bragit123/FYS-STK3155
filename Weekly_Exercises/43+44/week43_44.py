
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
from NN import FFNN
from scheduler import Constant

def sigmoid(X):
    return 1.0 / (1 + jnp.exp(-X))

def CostOLS(target):
    def func(X):
        return (1.0 / target.shape[0]) * jnp.sum((target - X)**2)
    return func

def CostCrossEntropy(target):
    
    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func

eta = 0.01
scheduler = Constant(eta)


X = np.array([[0,0,1,1],[0,1,0,1]]).T

t_XOR = np.array([0,1,1,0])
t_AND = np.array([0,0,0,1])
t_OR = np.array([0,1,1,1])
t_XOR = np.c_[t_XOR]
t_AND = np.c_[t_AND]
t_OR = np.c_[t_OR]

dim = (2, 2, 1)

Neural = FFNN(dim, sigmoid, CostOLS)

output = Neural.predict(X)
print("Before training:")
print(output)

Neural.train(X, t_XOR, scheduler)
output = Neural.predict(X)
print("After training:")
print(output)