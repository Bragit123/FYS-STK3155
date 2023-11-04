
import numpy as np
# import jax.numpy as jnp
import autograd.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from NN import FFNN
# from NN_new import FFNN
# from lec_notes import FFNN
from scheduler import Constant, Adam
from funcs import CostCrossEntropy, sigmoid
from copy import copy
from sklearn.neural_network import MLPClassifier

eta = 0.1
rho = 0.9
rho2 = 0.999
scheduler = Constant(eta)
scheduler = Adam(eta, rho, rho2)

X = np.array([[0,0,1,1],[0,1,0,1]]).T
# X = np.array([[0,1]], dtype=float)

t_XOR = np.array([0,1,1,0], dtype=float)
t_AND = np.array([0,0,0,1], dtype=float)
t_OR = np.array([0,1,1,1], dtype=float)
t_XOR = np.c_[t_XOR]
t_AND = np.c_[t_AND]
t_OR = np.c_[t_OR]

dim = (2, 2, 1)

Neural = FFNN(dim, act_func=sigmoid, cost_func=CostCrossEntropy, seed=100)

output = Neural.predict(X)
print("Before backpropagation")
print(output)

scores = Neural.train(X, t_XOR, scheduler, epochs=100)

output = Neural.predict(X)
print("After backpropagation")
print(output)

# print("Scores")
# print(scores)

# With scikit-learn
t_XOR = np.array([0,1,1,0], dtype=float)
clf = MLPClassifier(solver="sgd", alpha=0, hidden_layer_sizes=(2), random_state=1)
clf.fit(X, t_XOR)
output = clf.predict(X)
print("Scikit-learn")
print(output)