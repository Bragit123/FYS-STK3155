
import numpy as np
# import jax.numpy as jnp
import autograd.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from NN import FFNN
from lec_notes import FFNN
from scheduler import *
from funcs import CostCrossEntropy, sigmoid
from sklearn.utils import resample
import sklearn.model_selection

# def sigmoid(X):
#     return 1.0 / (1 + jnp.exp(-X))

# def CostOLS(target):
#     def func(X):
#         return (1.0 / target.shape[0]) * jnp.sum((target - X)**2)
#     return func

# def CostCrossEntropy(target):

#     def func(X):
#         return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

#     return func

eta = 0.01
rho = 0.9
rho2 = 0.999
scheduler = Adam(eta, rho, rho2)


X = np.array([[0,0,1,1],[0,1,0,1]]).T

t_XOR = np.array([0,1,1,0])
t_AND = np.array([0,0,0,1])
t_OR = np.array([0,1,1,1])
t_XOR = np.c_[t_XOR]
t_AND = np.c_[t_AND]
t_OR = np.c_[t_OR]

dim = (2, 2, 1)

# Neural = FFNN(dim, sigmoid, CostOLS)
Neural = FFNN(dim, output_func=sigmoid, cost_func=CostCrossEntropy, seed=100)
Neural.reset_weights()

output = Neural.predict(X)
print("Before training:")
print(output)

w = Neural.weights
# print("Weights:")
# print(w)

outputs = list()


def f(x):
    return 5+3*x+1*x**2
x = np.linspace(0,10,100)
y = f(x)
#beta = np.array([5,3,1])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y)
input_nodes = x_train.shape[0]
output_nodes = 1
linear_regression = FFNN((input_nodes, output_nodes), seed=2023)
linear_regression.reset_weights() # reset weights such that previous runs or reruns don't affect the weights
scheduler = AdagradMomentum(eta=1e-3,momentum=0.1) #eta
output = linear_regression.predict(x)
scores = linear_regression.fit(x_train, y_train, scheduler, epochs=1000)
#print(output)
plt.plot(x,y,"ro")
plt.plot(x,output[0][:-1],"yo")
plt.show()

eta = np.logspace(-5, 0, 6)
lam = np.logspace(-5, 0, 6)
res = np.zeros((len(eta), len(lam)))
for i in range(len(eta)):
    for j in range(len(lam)):
        Neural = FFNN(dim, output_func=sigmoid, cost_func=CostCrossEntropy, seed=123)
        Neural.reset_weights()
        rho = 0.9
        rho2 = 0.999
        scheduler = Adam(eta[i], rho, rho2)
        scores = Neural.fit(X, t_OR, scheduler, epochs=1000, lam=lam[j])
        output = Neural.predict(X)
        #res[i,j] = scores["train_accs"][-1]
        print(output)

sns.set()
fig, ax = plt.subplots(figsize = (10,10))
sns.heatmap(res, annot=True, ax=ax, cmap="viridis")
ax.set_title("Test_accuracy")
ax.set_ylabel("$\lambda$")
ax.set_xlabel("$\eta$")
plt.show()
# scores = Neural.fit(X, t_OR, scheduler, epochs=10000, lam=0.01)
# output = Neural.predict(X)
# print("After training:")
# print(output)
