
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import minmax_scale, LabelBinarizer
from sklearn.model_selection import train_test_split
import NN
from scheduler import AdamMomentum
from funcs import sigmoid, RELU, identity, CostLogReg, softmax, LRELU

from tensorflow.keras import datasets

(X_train, t_train), (X_test, t_test) = datasets.mnist.load_data()

t_train = LabelBinarizer().fit_transform(t_train)
t_test = LabelBinarizer().fit_transform(t_test)

## Reshape X to 1D
n_train, n_rows, n_cols = np.shape(X_train)
n_test, n_rows, n_cols = np.shape(X_test)
n_features = n_rows*n_cols
X_train = np.reshape(X_train, (n_train, n_features))
X_test = np.reshape(X_test, (n_test, n_features))

X_train = minmax_scale(X_train, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems
X_test = minmax_scale(X_test, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems

## Neural network
dim = (n_features, 100, 10)
hidden_act = sigmoid ; output_act = sigmoid
cost_func = CostLogReg
eta = 0.01 ; rho = 0.9 ; rho2 = 0.999 ; momentum = 0.01
scheduler = AdamMomentum(eta, rho, rho2, momentum)

n_epochs = 10
n_batches = 1
neural = NN.FFNN(dim, hidden_act, output_act, cost_func, categorization=True)
scores = neural.train(X_train, t_train, scheduler, epochs=n_epochs, batches=n_batches, X_val=X_test, t_val=t_test)

epochs = np.arange(n_epochs)
val_accs = scores["val_accs"]

plt.figure()
plt.plot(epochs, val_accs)
plt.savefig("accs.pdf")