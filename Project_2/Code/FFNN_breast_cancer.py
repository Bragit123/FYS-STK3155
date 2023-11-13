import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from funcs import identity, sigmoid, RELU, LRELU, CostLogReg
from scheduler import Adam, AdamMomentum
from NN import FFNN
from sklearn import datasets
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
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
dims = [
    (n_inputs, 100, n_outputs),
    (n_inputs, 50, n_outputs),
    (n_inputs, 10, n_outputs),
    (n_inputs, 50, 50, n_outputs),
    (n_inputs, 10, 10, n_outputs)
]
hidden_acts = [identity, sigmoid, RELU, LRELU]

# Make arrays for eta and lambda
eta0 = -4; eta1 = -1; n_eta = eta1-eta0+1
lam0 = -5; lam1 = -1; n_lam = lam1-lam0+1
etas = np.logspace(eta0, eta1, n_eta)
lams = np.logspace(lam0, lam1, n_lam)


# Create and train neural network for
for d in range(len(dims)):
    for a in range(len(hidden_acts)):
        train_errors = np.zeros((n_eta, n_lam))
        val_errors = np.zeros((n_eta, n_lam))
        train_accs = np.zeros((n_eta, n_lam))
        val_accs = np.zeros((n_eta, n_lam))
        for i in range(len(etas)):
            for j in range(len(lams)):
                dim = dims[d]
                hidden_act = hidden_acts[a]
                eta = etas[i]
                lam = lams[j]
                network = FFNN(dim, hidden_act, output_act, cost_func, seed, classification)
                output = network.predict(X_train)
                scheduler = AdamMomentum(eta, rho, rho2, momentum)
                scores = network.train(X_train, t_train, scheduler, batches, epochs, lam, X_val, t_val)
                train_errors[i,j] = scores["train_errors"][-1]
                val_errors[i,j] = scores["val_errors"][-1]
                train_accs[i,j] = scores["train_accs"][-1]
                val_accs[i,j] = scores["val_accs"][-1]

        # Format filename
        n_nodes = dim[1:-1]
        node_txt = ""
        for n_node in n_nodes:
            node_txt = node_txt + str(n_node) + "_"
        node_txt = node_txt[:-1]
        act_txt = hidden_act.__name__

        # Create heatmap plots for the accuracies
        title = f"Training accuracy on breast cancer dataset. Hidden activation function: {act_txt}. Dimension {dim}."
        filename = f"../Figures/Breast_cancer/train_accs_{act_txt}_{node_txt}.pdf"
        plotting.heatmap(data=train_accs, xticks=lams, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)

        title = f"Validation accuracy on breast cancer dataset. Hidden activation function: {act_txt}. Dimension {dim}."
        filename = f"../Figures/Breast_cancer/val_accs_{act_txt}_{node_txt}.pdf"
        plotting.heatmap(data=val_accs, xticks=lams, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)
