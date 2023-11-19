import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from funcs import identity, sigmoid, RELU, LRELU, CostLogReg
from scheduler import Adam, AdamMomentum
from NN import FFNN
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import plotting

# Load cancer dataset
cancer = datasets.load_breast_cancer()
X = cancer.data
X = X
t = cancer.target
t = np.c_[t]

X = minmax_scale(X, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems

X_train, X_val, t_train, t_val = train_test_split(X, t, test_size=0.2) # Split into training and validation sets

# Set values for neural network
output_act = sigmoid
hidden_act = sigmoid
cost_func = CostLogReg
seed = 100
classification = True

# Set constants
rho = 0.9
rho2 = 0.999
momentum = 0.01
batches = 1
epochs = 50

# Make lists for dimension and activation functions
n_inputs = 30; n_outputs = 1
dim = (30, 1)

# Make arrays for eta and lambda
eta0 = -4; eta1 = -1; n_eta = eta1-eta0+1
lam0 = -5; lam1 = -1; n_lam = lam1-lam0+1
etas = np.logspace(eta0, eta1, n_eta)
lams = np.logspace(lam0, lam1, n_lam)


# Create and train neural network for
val_accs_NN = np.zeros((n_eta, n_lam))
val_accs_scikit = np.zeros((n_eta, n_lam))
for i in range(len(etas)):
    for j in range(len(lams)):
        eta = etas[i]
        lam = lams[j]

        ## Our code
        network = FFNN(dim, hidden_act, output_act, cost_func, seed, classification)
        output = network.predict(X_train)
        scheduler = AdamMomentum(eta, rho, rho2, momentum)
        scores = network.train(X_train, t_train, scheduler, batches, epochs, lam, X_val, t_val)
        val_accs_NN[i,j] = scores["val_accs"][-1]

## Scikit-learn
Logreg = LogisticRegression()
Logreg.fit(X_train, t_train)
val_accs_scikit = Logreg.score(X_val, t_val)
print(f"Accuracy from scikit: {val_accs_scikit*100:.0f}%")


# Create heatmap plots for the accuracies
title = f"Validation accuracies from logistic regression."
filename = f"../Figures/Breast_cancer/val_accs_NN_logreg.pdf"
plotting.heatmap(data=val_accs_NN, xticks=lams, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)