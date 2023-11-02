import numpy as np
from typing import Callable
import jax.numpy as jnp
from jax import grad, jit, vmap, random
from GradientDescentSolver import *

key = random.PRNGKey(456)
class NeuralNetwork:
    def __init__(self, X, y, cost_func, deg, target, eta = 0.1,lmbd=0.01, n_hidden_neurons = 2, n_categories = 1, n_features = 2, gradient_func=None):
        self.X = X
        self.y = y
        self.cost_func = cost_func
        self.deg = deg
        self.beta = None
        if gradient_func == None:
            self.gradient_func = grad(cost_func)
        else:
            self.gradient_func = gradient_func

        self.tuning_algorithm = Tuning_algorithm() # Index of 'active' tuning algorithm (chosen from tuning_algorithms)
        # Defining the neural network
        self.n_inputs, self.n_features = X.shape
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories  #Only one output node
        self.n_features = n_features

        # weights and bias in the hidden layer
        self.hidden_weights = random.normal(key,shape=(n_features,n_hidden_neurons))
        self.hidden_bias = jnp.zeros(n_hidden_neurons) + 0.01

        # weights and bias in the output layer
        self.output_weights = random.normal(key,shape=(n_hidden_neurons,n_categories))
        self.output_bias = jnp.zeros(n_categories) + 0.01


    def feed_forward(self,X):
        # weighted sum of inputs to the hidden layer
        z_h = np.matmul(X, hidden_weights) + hidden_bias
        # activation in the hidden layer
        a_h = sigmoid(z_h)

        # weighted sum of inputs to the output layer
        z_o = np.matmul(a_h, output_weights) + output_bias
        # softmax output
        # axis 0 holds each input and axis 1 the probabilities of each category
        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

        return a_h, probabilities

    # we obtain a prediction by taking the class with the highest likelihood
    def predict(self,X):
        probabilities = feed_forward(X)[1]
        return np.argmax(probabilities, axis=1)

    def backpropagate(self,n_iter):
        for i in range(n_iter):
            a_h, probabilities = feed_forward(X)
            target = self.target.reshape(-1,1)
            error_output = probabilities - target_XOR #grad_func(probabilities)
            error_hidden = jnp.matmul(error_output, self.output_weights.T) * a_h * (1 - a_h)

            # gradients for the output layer
            output_weights_gradient = jnp.matmul(a_h.T, error_output)
            output_bias_gradient = jnp.sum(error_output, axis=0)

            # gradient for the hidden layer
            hidden_weights_gradient = jnp.matmul(X.T, error_hidden)
            hidden_bias_gradient = jnp.sum(error_hidden, axis=0)

            #Regularization terms gradients
            output_weights_gradient += self.lmbd * output_weights
            hidden_weights_gradient += self.lmbd * hidden_weights

            # weights and bias in the hidden layer
            self.hidden_weights -= self.eta*hidden_weights_gradient
            self.hidden_bias -= self.eta*hidden_bias_gradient

            # weights and bias in the output layer
            self.output_weights -= eta*output_weights_gradient
            self.output_bias -= eta*output_bias_gradient
        return 0
