import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from funcs import sigmoid, RELU, LRELU, CostCrossEntropy, CostLogReg
from scheduler import Adam, AdamMomentum
from NN import FFNN
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split

# Load cancer dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
X = X[:, :10]
t = cancer.target
t = np.c_[t]

X = minmax_scale(X, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems

X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2)

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
output = network.predict(X_train)
scores = network.train(X_train, t_train, scheduler, batches, epochs, lam, X_val, t_val)

# Plot error
epoch_arr = np.arange(epochs)
plt.figure()
plt.title("Training errors")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.plot(epoch_arr, scores["train_errors"], label="Train error")
plt.plot(epoch_arr, scores["val_errors"], label="Validation error")
plt.legend()
plt.savefig("error.pdf")

plt.figure()
plt.title("Training errors")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.plot(epoch_arr, scores["train_accs"], label="Train accuracy")
plt.plot(epoch_arr, scores["val_accs"], label="Validation accuracy")
plt.legend()
plt.savefig("accuracy.pdf")