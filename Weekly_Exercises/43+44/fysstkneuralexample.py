#Code taken from example
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import jax.numpy as jnp
from jax import grad, random, jit

def sigmoid(x):
    return 1.0/(1.0 + jnp.exp(-x))

def sigmoid_grad(x):
    return jnp.mean(1.0/(1.0 + jnp.exp(-x)))
#def grad_sigmoid(x):
#    def func(x):
#        return 1/(1 + jnp.exp(-x))
#    return func


def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    z_h = jnp.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)

    # weighted sum of inputs to the output layer
    z_o = np.matmul(a_h, output_weights) + output_bias
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    probabilities = sigmoid(z_o)
    return probabilities

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    probabilities = feed_forward(X)
    return jnp.argmax(probabilities, axis=1)

# ensure the same random numbers appear every time
key = random.PRNGKey(123)

# Design matrix
X = jnp.array([ [0.0, 0.0], [0.0, 1.0], [1.0, 0.0],[1.0, 1.0]])

# The XOR gate
yXOR = jnp.array( [ 0.0, 1.0 ,1.0, 0.0])
# The OR gate
yOR = jnp.array( [ 0.0, 1.0 ,1.0, 1.0])
# The AND gate
yAND = jnp.array( [ 0.0, 0.0 ,0.0, 1.0])

# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 2
n_features = 2

# we make the weights normally distributed using random.normal from jax

# weights and bias in the hidden layer
hidden_weights = random.normal(key,shape=(n_features,n_hidden_neurons))
hidden_bias = jnp.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = random.normal(key,shape=(n_hidden_neurons,n_categories))
output_bias = jnp.zeros(n_categories) + 0.01

probabilities = feed_forward(X)
print(probabilities)


predictions = predict(X)
print(predictions)


# Defining the neural network
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 1  #Only one output node
n_features = 2



# weights and bias in the hidden layer
hidden_weights = random.normal(key,shape=(n_features,n_hidden_neurons))
hidden_bias = jnp.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = random.normal(key,shape=(n_hidden_neurons,n_categories))
output_bias = jnp.zeros(n_categories) + 0.01

probabilities = feed_forward(X)
print(probabilities)


predictions = predict(X)
print(predictions)

#Setting up the cost function
def CostCrossEntropy(target):

    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func

"""
def gradCostCrossEntropy(target,X):

    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return grad(func)
"""
target_XOR = yXOR.reshape(-1,1)
target_OR = yOR
target_AND = yAND

cost_func = CostCrossEntropy
grad_func = grad(cost_func(target_XOR))
grad_sigmoid = grad(sigmoid_grad)
eta = 0.1


# Defining the neural network, not sure about these
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 1
n_features = 2

# weights and bias in the hidden layer
hidden_weights = random.normal(key,shape=(n_features,n_hidden_neurons))
hidden_bias = jnp.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = random.normal(key,shape=(n_hidden_neurons,n_categories))
output_bias = jnp.zeros(n_categories) + 0.01

def feed_forward(X):
    # weighted sum of inputs to the hidden layer
    z_h = jnp.matmul(X, hidden_weights) + hidden_bias
    # activation in the hidden layer
    a_h = sigmoid(z_h)

    # weighted sum of inputs to the output layer
    z_o = jnp.matmul(a_h, output_weights) + output_bias
    # softmax output
    # axis 0 holds each input and axis 1 the probabilities of each category
    probabilities = sigmoid(z_o)
    return a_h, z_o, z_h, probabilities

# we obtain a prediction by taking the class with the highest likelihood
def predict(X):
    probabilities = feed_forward(X)[1]
    return jnp.argmax(probabilities, axis=1)

lmbd = 0.01
#XOR, back propagation
for i in range(1000):
    a_h, z_o,z_h, probabilities = feed_forward(X)
    target_XOR = target_XOR.reshape(-1,1)
    #z_o.reshape((z_o.shape[0],))
    #print(z_o.shape)

    error_output = grad_func(probabilities)*grad_sigmoid(z_o) #probabilities-target_XOR
    #print(a_h * (1 - a_h))
    #print(grad_sigmoid(z_h))
    error_hidden = jnp.matmul(error_output, output_weights.T)*grad_sigmoid(z_h)#** a_h * (1 - a_h)

    # gradients for the output layer
    output_weights_gradient = jnp.matmul(a_h.T, error_output)
    output_bias_gradient = jnp.sum(error_output, axis=0)

    # gradient for the hidden layer
    hidden_weights_gradient = jnp.matmul(X.T, error_hidden)
    hidden_bias_gradient = jnp.sum(error_hidden, axis=0)

    #Regularization terms gradients
    output_weights_gradient += lmbd * output_weights
    hidden_weights_gradient += lmbd * hidden_weights

    # weights and bias in the hidden layer
    hidden_weights -= eta*hidden_weights_gradient
    hidden_bias -= eta*hidden_bias_gradient

    # weights and bias in the output layer
    output_weights -= eta*output_weights_gradient
    output_bias -= eta*output_bias_gradient


a_h,z_o,z_h,predictions = feed_forward(X)
print("XOR result after training")
print(predictions)



cost_func = CostCrossEntropy
grad_func = grad(cost_func(target_AND))
grad_sigmoid = grad(sigmoid_grad)
eta = 0.1

# Defining the neural network, not sure about these
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 1
n_features = 2

# weights and bias in the hidden layer
hidden_weights = random.normal(key,shape=(n_features,n_hidden_neurons))
hidden_bias = jnp.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = random.normal(key,shape=(n_hidden_neurons,n_categories))
output_bias = jnp.zeros(n_categories) + 0.01

lmbd = 0.01
#AND, back propagation
for i in range(1000):
    a_h, z_o, probabilities = feed_forward(X)
    target_AND = target_AND.reshape(-1,1)
    #z_o.reshape((z_o.shape[0],))
    #print(z_o.shape)
    error_output = grad_func(probabilities)*grad_sigmoid(z_o) #probabilities-target_XOR
    error_hidden = jnp.matmul(error_output, output_weights.T)*grad_sigmoid(z_o)#* a_h * (1 - a_h)

    # gradients for the output layer
    output_weights_gradient = jnp.matmul(a_h.T, error_output)
    output_bias_gradient = jnp.sum(error_output, axis=0)

    # gradient for the hidden layer
    hidden_weights_gradient = jnp.matmul(X.T, error_hidden)
    hidden_bias_gradient = jnp.sum(error_hidden, axis=0)

    #Regularization terms gradients
    output_weights_gradient += lmbd * output_weights
    hidden_weights_gradient += lmbd * hidden_weights

    # weights and bias in the hidden layer
    hidden_weights -= eta*hidden_weights_gradient
    hidden_bias -= eta*hidden_bias_gradient

    # weights and bias in the output layer
    output_weights -= eta*output_weights_gradient
    output_bias -= eta*output_bias_gradient


a_h,z_o,predictions = feed_forward(X)
print("AND result after training")
print(predictions)




cost_func = CostCrossEntropy
grad_func = grad(cost_func(target_OR))
grad_sigmoid = grad(sigmoid_grad)
eta = 0.1

# Defining the neural network, not sure about these
n_inputs, n_features = X.shape
n_hidden_neurons = 2
n_categories = 1
n_features = 2

# weights and bias in the hidden layer
hidden_weights = random.normal(key,shape=(n_features,n_hidden_neurons))
hidden_bias = jnp.zeros(n_hidden_neurons) + 0.01

# weights and bias in the output layer
output_weights = random.normal(key,shape=(n_hidden_neurons,n_categories))
output_bias = jnp.zeros(n_categories) + 0.01

lmbd = 0.01
#OR, back propagation
for i in range(1000):
    a_h, z_o, probabilities = feed_forward(X)
    target_OR = target_OR.reshape(-1,1)
    z_o.reshape((z_o.shape[0],))
    #print(z_o.shape)
    error_output = grad_func(probabilities)*grad_sigmoid(z_o) #probabilities-target_XOR
    error_hidden = jnp.matmul(error_output, output_weights.T)*grad_sigmoid(z_o)#* a_h * (1 - a_h)

    # gradients for the output layer
    output_weights_gradient = jnp.matmul(a_h.T, error_output)
    output_bias_gradient = jnp.sum(error_output, axis=0)

    # gradient for the hidden layer
    hidden_weights_gradient = jnp.matmul(X.T, error_hidden)
    hidden_bias_gradient = jnp.sum(error_hidden, axis=0)

    #Regularization terms gradients
    output_weights_gradient += lmbd * output_weights
    hidden_weights_gradient += lmbd * hidden_weights

    # weights and bias in the hidden layer
    hidden_weights -= eta*hidden_weights_gradient
    hidden_bias -= eta*hidden_bias_gradient

    # weights and bias in the output layer
    output_weights -= eta*output_weights_gradient
    output_bias -= eta*output_bias_gradient


a_h,z_o,predictions = feed_forward(X)
print("OR result after training")
print(predictions)
