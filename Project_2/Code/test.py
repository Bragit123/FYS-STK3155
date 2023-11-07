import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from funcs import sigmoid, RELU, LRELU, CostCrossEntropy, CostLogReg
from scheduler import Adam, AdamMomentum
from NN import FFNN
from sklearn import datasets
from sklearn.preprocessing import minmax_scale

# Load cancer dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
X = X[:, :10]
t = cancer.target
t = np.c_[t]

X = minmax_scale(X, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems

# Set values for neural network
dim = (10, 100, 1)
hidden_act = sigmoid
output_act = sigmoid
cost_func = CostLogReg
seed = 500
classification = True

eta = 0.01
rho = 0.9
rho2 = 0.999
momentum = 0.01
scheduler = AdamMomentum(eta, rho, rho2, momentum)
batches = 1
epochs = 100
lam = 0

# Create and train neural network
network = FFNN(dim, hidden_act, output_act, cost_func, seed, classification)
output = network.predict(X)
scores = network.train(X, t, scheduler, batches, epochs, lam)
output = network.predict(X)

# Plot error
epoch_arr = np.arange(epochs)
plt.title("Training errors")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.plot(epoch_arr, scores["train_errors"], label="Error")
plt.plot(epoch_arr, scores["train_accs"], label="Accuracy")
plt.legend()
plt.savefig("train_err.pdf")

print(scores["train_errors"])
print(scores["train_accs"])
