
import numpy as np
from sklearn.preprocessing import minmax_scale, LabelBinarizer
from NN import FFNN
from scheduler import AdamMomentum
from funcs import sigmoid, RELU, identity, CostLogReg, LRELU
import plotting
from tensorflow.keras import datasets
seed = 100

# Load MNIST dataset
(X_train, t_train), (X_test, t_test) = datasets.mnist.load_data()

# Retrieve only a subset for faster computation
n_train = int(0.1*len(t_train))
n_test = int(0.1*len(t_test))
X_train = X_train[0:n_train,:,:] ; t_train = t_train[0:n_train]
X_test = X_test[0:n_test,:,:] ; t_test = t_test[0:n_test]

# Reshape target data to fit with the output of our neural network (10 nodes,
# where the node corresponding to the right digit is 1, and the rest is 0).
t_train = LabelBinarizer().fit_transform(t_train)
t_test = LabelBinarizer().fit_transform(t_test)

# Reshape data to 1D
n_train, n_rows, n_cols = np.shape(X_train)
n_test, n_rows, n_cols = np.shape(X_test)
n_features = n_rows*n_cols
X_train = np.reshape(X_train, (n_train, n_features))
X_test = np.reshape(X_test, (n_test, n_features))

# Scale data
X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems
X_test = minmax_scale(X_test, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems

# Define some constants for the neural network
dim = (n_features, 50, 10)
output_act = sigmoid
cost_func = CostLogReg
rho = 0.9 ; rho2 = 0.999 ; momentum = 0.01

n_epochs = 50
batch_size = 400
n_batches = X_train.shape[0] // batch_size


# Make arrays for hidden activation functions, etas and lambdas
hidden_acts = [identity, sigmoid, RELU, LRELU]
eta0 = -5; eta1 = 1; n_eta = eta1-eta0+1
lam0 = -5; lam1 = 1; n_lam = lam1-lam0+1
etas = np.logspace(eta0, eta1, n_eta)
lams = np.logspace(lam0, lam1, n_lam)

# Create and train neural network for different hidden activation functions,
# etas and lambdas
for a in range(len(hidden_acts)):
    hidden_act = hidden_acts[a]
    val_accs = np.zeros((n_eta, n_lam))
    for i in range(len(etas)):
        for j in range(len(lams)):
            eta = etas[i]
            lam = lams[j]
            network = FFNN(dim, hidden_act, output_act, cost_func, seed, categorization=True)
            output = network.predict(X_train)
            scheduler = AdamMomentum(eta, rho, rho2, momentum)
            scores = network.train(X_train, t_train, scheduler, n_batches, n_epochs, lam, X_test, t_test)
            val_accs[i,j] = scores["val_accs"][-1]

    # Format filename
    act_txt = hidden_act.__name__

    # Create heatmap plots for the accuracies
    title = f"Hidden activation function: {act_txt}."
    filename = f"../Figures/val_accs_NN_{act_txt}.pdf"
    plotting.heatmap(data=val_accs, xticks=lams, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)
