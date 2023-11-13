import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from funcs import sigmoid, RELU, LRELU, CostCrossEntropy, CostLogReg
from scheduler import Adam, AdamMomentum
from NN import FFNN
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
import plotting

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
seed = 100
classification = True

eta = 0.01
rho = 0.9
rho2 = 0.999
momentum = 0.01
scheduler = AdamMomentum(eta, rho, rho2, momentum)
batches = 1
epochs = 50
lam = 0

# Create and train neural network
network = FFNN(dim, hidden_act, output_act, cost_func, seed, classification)
output = network.predict(X_train)
scores = network.train(X_train, t_train, scheduler, batches, epochs, lam, X_val, t_val)


# Plot error and accuracy
epoch_arr = np.arange(epochs)

x_plot = [epoch_arr, epoch_arr]
y_plot = [scores["train_errors"], scores["val_errors"]]
labels = ["Train error", "Validation error"]
title = "Error of neural network model on breast cancer dataset."
xlabel = "Epoch"; ylabel = "Value"
filename = "error.pdf"

plotting.plot(x_plot, y_plot, labels, title, xlabel, ylabel, filename)

labels = ["Train accuracy", "Validation accuracy"]
title = "Accuracy of neural network model on breast cancer dataset."
y_plot = [scores["train_accs"], scores["val_accs"]]
filename = "accuracy.pdf"
plotting.plot(x_plot, y_plot, labels, title, xlabel, ylabel, filename)

print(scores["train_errors"][-1])
print(scores["val_errors"][-1])
print(scores["train_accs"][-1])
print(scores["val_accs"][-1])