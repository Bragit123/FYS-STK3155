# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from NN import *
from scheduler import *
from funcs import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
# ensure the same random numbers appear every time
np.random.seed(0)
a = np.array([[1,2,3],[4,5,6],[]])
print()

# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target
labels = LabelBinarizer().fit_transform(labels)
#print(labels)
# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64
inputs_shape = len(inputs)
inputs = inputs.reshape(inputs_shape, -1)
n_inputs = len(inputs[0])
print(n_inputs)
print("X = (n_inputs, n_features) = " + str(inputs.shape))


# choose some random images to display
#indices = np.arange(n_inputs)
#random_indices = np.random.choice(indices, size=5)

#for i, image in enumerate(digits.images[random_indices]):
#    plt.subplot(1, 5, i+1)


plt.axis('off')
plt.imshow(digits.images[3], cmap=plt.cm.gray_r, interpolation='nearest')
plt.title("Label: %d" % digits.target[3])
plt.show()


# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)
#Y_train = Y_train.reshape(-1,1)
#Y_test = Y_test.reshape(-1,1)
#print(Y_train.shape)
#X_train, X_test, Y_train, Y_test = train_test_split_numpy(inputs, labels, train_size, test_size)

print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))


# Set values for neural network
output_act = identity
cost_func = CostLogReg
seed = 100
classification = False

# Set constants
rho = 0.9
rho2 = 0.999
momentum = 0.01
batches = 1
epochs = 50

# Make lists for dimension and activation functions
#n_inputs = 30; n_outputs = 1
n_outputs = 1
print(n_inputs)
dims = (n_inputs, 50, n_outputs)
hidden_act = RELU

eta = 1e-2
lmbd = 1e-2

network = FFNN(dims, hidden_act, output_act, cost_func, seed, classification)
#output = network.predict(X_train)
scheduler = AdamMomentum(eta, rho, rho2, momentum)
scores = network.train(X_train, Y_train, scheduler, batches, epochs, lmbd, X_test, Y_test)
#train_error = scores["train_errors"][-1]
#val_error = scores["val_errors"][-1]
#train_acc = scores["train_accs"][-1]
#val_acc = scores["val_accs"][-1]
#print(inputs[2:7])
output_test = network.predict(inputs[2:7])
print(labels[2:7], " ", output_test)

#print(val_acc)
# Make arrays for eta and lambda
"""
eta0 = -4; eta1 = -1; n_eta = eta1-eta0+1
lam0 = -5; lam1 = -1; n_lam = lam1-lam0+1
etas = np.logspace(eta0, eta1, n_eta)
lams = np.logspace(lam0, lam1, n_lam)
"""
"""
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
        title = f"Hidden activation function: {act_txt}. Dimension {dim}."
        filename = f"../Figures/Breast_cancer/train_accs_{act_txt}_{node_txt}.pdf"
        plotting.heatmap(data=train_accs, xticks=lams, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)

        title = f"Hidden activation function: {act_txt}. Dimension {dim}."
        filename = f"../Figures/Breast_cancer/val_accs_{act_txt}_{node_txt}.pdf"
        plotting.heatmap(data=val_accs, xticks=lams, yticks=etas, title=title, xlabel="$\\lambda$", ylabel="$\\eta$", filename=filename)
"""
