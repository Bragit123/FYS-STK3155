import numpy as np
from jax import jacobian, vmap, grad
import jax as jnp
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import minmax_scale, LabelBinarizer
from sklearn.model_selection import train_test_split
import NN
from scheduler import AdamMomentum
from funcs import sigmoid, RELU, identity, CostLogReg, softmax, LRELU

from tensorflow.keras import datasets



# Jacobian of softmax for a single row
jacobian_softmax_single = jax.jacfwd(softmax)

# Vectorized Jacobian using vmap
jacobian_act_output = vmap(jacobian_softmax_single)

# Example usage with a matrix
input_matrix = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

# Apply vmap to calculate Jacobian for each row
jacobian_matrix = jacobian_act_output(input_matrix)

# Take the mean along the third axis to obtain a single 3x3 Jacobian matrix
mean_jacobian_matrix = jnp.mean(jacobian_matrix, axis=2)

print(mean_jacobian_matrix)
digits = load_digits()

X = digits.images
t = digits.target


t = LabelBinarizer().fit_transform(t)
n_inputs, n_rows, n_cols = np.shape(X)
n_features = n_rows*n_cols
X = np.reshape(X, (n_inputs, n_features))

X = minmax_scale(X, feature_range=(0, 1), axis=0) # Scale to avoid sigmoid problems

X_train, X_test, t_train, t_test = train_test_split(X, t, test_size=0.2, random_state=100)

## Neural network
dim = (n_features, 100, 10)
hidden_act = sigmoid ; output_act = identity
cost_func = CostLogReg
eta = 0.01 ; rho = 0.9 ; rho2 = 0.999 ; momentum = 0.01 ; lmbd = 0.001
scheduler = AdamMomentum(eta, rho, rho2, momentum)
batches = 20
epochs = 100

neural = NN.FFNN(dim, hidden_act, output_act, cost_func, categorization=True)
scores = neural.train(X_train, t_train, scheduler, epochs=500, X_val=X_test, t_val=t_test)

epochs = np.arange(500)
val_accs = scores["val_accs"]

plt.figure()
plt.plot(epochs, val_accs)
plt.savefig("accs.pdf")
