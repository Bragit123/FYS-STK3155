import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import sklearn
from sklearn.model_selection import train_test_split


def FrankeFunction(x, y):
    """ Calculates the Franke function at a point (x,y) """
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

x = np.arange(0, 1, 0.05)
y = np.arange(0, 1, 0.05)
x, y = np.meshgrid(x,y)
x_shape = x.shape
y_shape = y.shape
x = x.flatten()
y = y.flatten()
X = np.array([x.flatten(),y.flatten()]).T
target = FrankeFunction(x.flatten(),y.flatten())
target += 0.1*np.random.normal(0, 1, target.shape) #Adding noise
target_shape = target.shape
X_train, X_test, t_train, t_test = train_test_split(X, target, test_size=0.2)
#RELU
# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 50
n_categories = 1
n_features = 2

eta_vals = np.logspace(-3, 0, 4)
lmbd_vals = np.logspace(-4, 0, 5)
# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
epochs = 100

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation="relu", solver="adam",
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs,batch_size=5, momentum=0.01)
        dnn.fit(X_train, t_train)
        DNN_scikit[i][j] = dnn
        #print("Learning rate  = ", eta)
        #print("Lambda = ", lmbd)
        #print("Accuracy score on data set: ", dnn.score(X, yXOR))
        #print()

sns.set()
MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        test_pred = dnn.predict(X_test)
        MSE[i,j] = sklearn.metrics.mean_squared_error(t_test,test_pred)
        R2[i,j] = sklearn.metrics.r2_score(t_test, test_pred)

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(MSE, xticklabels=lmbd_vals, yticklabels=eta_vals, annot=True, ax=ax, cmap="viridis")
ax.set_title("MSE test")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig("../Figures/ScikitMSErelu.pdf")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(R2, xticklabels=lmbd_vals, yticklabels=eta_vals, annot=True, ax=ax, cmap="viridis")
ax.set_title("R2-score test")
ax.set_ylabel("$\eta$")
ax.set_xlabel("$\lambda$")
plt.savefig("../Figures/ScikitR2relu.pdf")
plt.show()

heatmap(MSE, xticks=lmbds, yticks=etas, title="MSE test with scikit, RELU", xlabel="$\eta$", ylabel="$\lambda$", filename="../Figures/ScikitMSErelu.pdf")
heatmap(R2, xticks=lmbds, yticks=etas, title="R2-score with scikit, RELU", xlabel="$\eta$", ylabel="$\lambda$", filename="../Figures/ScikitR2relu.pdf")

#sigmoid
# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 50
n_categories = 1
n_features = 2

eta_vals = np.logspace(-3, 0, 4)
lmbd_vals = np.logspace(-4, 0, 5)
# store models for later use
DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
epochs = 100

for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        dnn = MLPRegressor(hidden_layer_sizes=(n_hidden_neurons), activation="relu", solver="adam",
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs,batch_size=5, momentum=0.01)
        dnn.fit(X_train, t_train)
        DNN_scikit[i][j] = dnn
        #print("Learning rate  = ", eta)
        #print("Lambda = ", lmbd)
        #print("Accuracy score on data set: ", dnn.score(X, yXOR))
        #print()

sns.set()
MSE = np.zeros((len(eta_vals), len(lmbd_vals)))
R2 = np.zeros((len(eta_vals), len(lmbd_vals)))
for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_scikit[i][j]
        test_pred = dnn.predict(X_test)
        MSE[i,j] = sklearn.metrics.mean_squared_error(t_test,test_pred)
        R2[i,j] = sklearn.metrics.r2_score(t_test, test_pred)

heatmap(MSE, xticks=lmbds, yticks=etas, title="MSE test with scikit, RELU", xlabel="$\eta$", ylabel="$\lambda$", filename="../Figures/ScikitMSErelu.pdf")
heatmap(R2, xticks=lmbds, yticks=etas, title="R2-score with scikit, RELU", xlabel="$\eta$", ylabel="$\lambda$", filename="../Figures/ScikitR2relu.pdf")
