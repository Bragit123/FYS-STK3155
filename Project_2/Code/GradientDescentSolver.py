import numpy as np
from typing import Callable
import jax.numpy as jnp
from jax import grad, jit, vmap
class Tuning_algorithm:
    def __init__(self):
        self.rho = 0.99
        self.beta1=0.9
        self.beta2=0.999
        self.delta=1e-8
        self.eta=0.01
        self.Giter=0.0
        self.first_moment = 0.0
        self.second_moment = 0.0
        self.algorithm = 0 #0: Default, 1:Adagrad, 2:RMS-prop, 3:ADAM

    def set_tuning_algorithm(self, algorithm_name: str):
        """ Choose algorithm for tuning the learning rate """
        algorithm_name = algorithm_name.upper()
        if algorithm_name == "DEFAULT":
            self.algorithm = 0
        elif algorithm_name == "ADAGRAD":
            self.algorithm = 1
        elif algorithm_name == "RMSPROP":
            self.algorithm = 2
        elif algorithm_name == "ADAM":
            self.algorithm = 3
        return 0

    def reset_params(self):
        self.Giter = 0
        self.first_moment = 0
        self.second_moment = 0

    def update(self, gradients: np.ndarray, iter: int = None):
        if self.algorithm == 0:
            update = self.eta*gradients


        elif self.algorithm == 1: #Adagrad
            Giter = (self.rho*self.Giter+(1-self.rho)*gradients*gradients)
            update = gradients*self.eta/(self.delta+np.sqrt(self.Giter))

        elif self.algorithm==2: #RMS-prop
            self.Giter = (self.rho*self.Giter+(1-rho)*gradients*gradients)
            update = gradients*self.eta/(self.delta+np.sqrt(self.Giter))

        elif self.algorithm == 3: #Adam
            if iter == None:
                print("Remember to set iter!")
            self.first_moment = self.beta1*self.first_moment + (1-self.beta1)*gradients
            self.second_moment = self.beta2*self.second_moment+(1-self.beta2)*gradients*gradients
            first_term = self.first_moment/(1.0-self.beta1**iter)
            second_term = self.second_moment/(1.0-self.beta2**iter)
            update = self.eta*self.first_term/(np.sqrt(self.second_term)+self.delta)
        return update


class Gradient_Descent:
    def __init__(self, X, y, cost_func, beta):
        self.X = X
        self.y = y
        self.cost_func = cost_func
        self.beta = beta
        self.gradient_func = None

        self.tuning_algorithm = Tuning_algorithm() # Index of 'active' tuning algorithm (chosen from tuning_algorithms)

    def analytic_gradient(self, gradient_func):
        """ Set an analytic gradient function to use in gradient descent
        methods."""
        self.gradient_func = gradient_func


    def learning_schedule(self, t0, t1, t):
        """ Compute learning rate (might want to have t0 and t1 as class
        attributes instead) """
        return t0/(t + t1)

    def GD(self, n_iter, momentum = None, use_jax = True, algorithm: str="DEFAULT"):
        """ Compute gradient descent (with momentum if specified)"""
        if use_jax == True:
            gradient_func = grad(self.cost_func)
        else:
            gradient_func = self.gradient_func
        self.tuning_algorithm.set_tuning_algorithm(algorithm)

        gradient = gradient_func(self.beta)
        if momentum == None:
            for iter in range(n_iter):
                # gradient = (2.0/n)*X.T @ (X @ beta-y)
                gradient = gradient_func(awlf.beta)
                update_beta = self.tuning_algorithm.update(gradient, iter)
                self.beta -= update_beta

        else:
            for iter in range(n_iter):
                mom_term = momentum*np.copy(gradient) #adding the momentum term using the previous gradient
                gradient = gradient_func(beta)
                update_beta = self.tuning_algorithm.update(gradient, iter)+mom_term
                self.beta -= update_beta

        self.tuning_algorithm.set_tuning_algorithm("DEFAULT")
        self.tuning_algorithm.reset_params()
        return self.beta

    def SGD(self, n_iter, n_epochs, batch_size, momentum = None, use_jax = True):
        M = batch_size
        n = len(self.y)
        m = int(n/M) #number of minibatches
        """ Compute stochastic gradient descent (with momentum if specified)"""
        if use_jax == True:
            gradient_func = grad(self.cost_func)
        else:
            gradient_func = self.gradient_func
        self.tuning_algorithm.set_tuning_algorithm(algorithm)
        if momentum == None:
            for epoch in range(n_epochs):
                self.tuning_algorithm.reset_params()
                for i in range(m):
                    random_index = M*np.random.randint(m)
                    xi = X[random_index:random_index+M]
                    yi = y[random_index:random_index+M]
                    gradient = (1.0/M)*gradient_func(yi, xi, beta)
                    update_beta = self.tuning_algorithm.update(gradient, iter)
                    self.beta -= update_beta

        else:
            for epoch in range(n_epochs):
                self.tuning_algorithm.reset_params()
                for i in range(m):
                    random_index = M*np.random.randint(m)
                    xi = X[random_index:random_index+M]
                    yi = y[random_index:random_index+M]
                    gradient = (1.0/M)*gradient_func(yi, xi, beta)
                    mom_term = momentum*np.copy(gradient)
                    update_beta = self.tuning_algorithm.update(gradient, iter) + mom_term
                    self.beta -= update_beta

        self.tuning_algorithm.set_tuning_algorithm("DEFAULT")

        return self.beta
