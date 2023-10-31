import numpy as np
from typing import Callable
import jax.numpy as jnp
from jax import grad, jit, vmap, random
key = random.PRNGKey(456)
class Tuning_algorithm:
    def __init__(self):
        self.rho = 0.99
        self.beta1=0.9
        self.beta2=0.999
        self.delta=1e-7
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

    def update(self, gradients: jnp.ndarray, iter: int = None):
        if self.algorithm == 0:
            update = self.eta*gradients


        elif self.algorithm == 1: #Adagrad
            self.Giter = self.Giter+gradients*gradients
            update = gradients*self.eta/(self.delta+jnp.sqrt(self.Giter))

        elif self.algorithm==2: #RMS-prop
            self.Giter = (self.rho*self.Giter+(1-self.rho)*gradients*gradients)
            update = gradients*self.eta/(self.delta+jnp.sqrt(self.Giter))

        elif self.algorithm == 3: #Adam
            if iter == None:
                print("Remember to set iter!")
            self.first_moment = self.beta1*self.first_moment + (1-self.beta1)*gradients
            self.second_moment = self.beta2*self.second_moment + (1-self.beta2)*gradients*gradients
            first_term = self.first_moment/(1.0-self.beta1**iter)
            second_term = self.second_moment/(1.0-self.beta2**iter)
            update = self.eta*first_term/(jnp.sqrt(second_term)+self.delta)
        return update


class Gradient_Descent:
    def __init__(self, X, y, cost_func, deg, gradient_func=None):
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

    def analytic_gradient(self, gradient_func):
        """ Set an analytic gradient function to use in gradient descent
        methods."""
        self.gradient_func = gradient_func

    def set_tuning_parameters(self,rho=0.99,beta1=0.9,beta2=0.999,delta = 1e-8,eta=0.01,Giter=0.0,first_moment=0.0,second_moment=0.0):
        self.tuning_algorithm.rho = rho
        self.tuning_algorithm.beta1=beta1
        self.tuning_algorithm.beta2=beta2
        self.tuning_algorithm.delta=delta
        self.tuning_algorithm.eta=eta
        self.tuning_algorithm.Giter=Giter
        self.tuning_algorithm.first_moment = first_moment
        self.tuning_algorithm.second_moment = second_moment
        return 0


    def learning_schedule(self, t0, t1, t):
        """ Compute learning rate (might want to have t0 and t1 as class
        attributes instead) """
        return t0/(t + t1)

    def GD(self, n_iter, momentum = None, algorithm: str="DEFAULT"):
        """ Compute gradient descent (with momentum if specified)"""
        self.beta = random.normal(key,shape=(self.deg+1,1))
        """
        if use_jax == True:
            gradient_func = grad(self.cost_func)
        else:
            gradient_func = self.gradient_func
            """
        self.tuning_algorithm.set_tuning_algorithm(algorithm)

        gradient = self.gradient_func(self.beta)
        if momentum == None:
            for iter in range(1,n_iter+1):
                # gradient = (2.0/n)*X.T @ (X @ beta-y)
                gradient = self.gradient_func(self.beta)
                update_beta = self.tuning_algorithm.update(gradient, iter)
                self.beta -= update_beta


        else:
            for iter in range(1,n_iter+1):
                mom_term = momentum*jnp.copy(gradient) #adding the momentum term using the previous gradient
                gradient = self.gradient_func(self.beta)
                update_beta = self.tuning_algorithm.update(gradient, iter) + mom_term
                self.beta -= update_beta


        self.tuning_algorithm.set_tuning_algorithm("DEFAULT")
        self.tuning_algorithm.reset_params()
        return self.beta

    def SGD(self, n_iter, n_epochs, batch_size, momentum = None,algorithm: str="DEFAULT"):
        self.beta = random.normal(key,shape=(self.deg+1,1))
        M = batch_size
        n = len(self.y)
        m = int(n/M) #number of minibatches
        """ Compute stochastic gradient descent (with momentum if specified)"""
        """
        if use_jax == True:
            gradient_func = grad(self.cost_func)
        else:
            gradient_func = self.gradient_func

            """
        self.tuning_algorithm.set_tuning_algorithm(algorithm)
        if momentum == None:
            for epoch in range(n_epochs):
                self.tuning_algorithm.reset_params()
                for iter in range(1,m+1):
                    #t = epoch*m + iter
                    #self.tuning_algorithm.eta = self.learning_schedule(5,50,t)
                    random_index = M*random.randint(key,shape=(1,),minval=0,maxval=m)[0]
                    xi = self.X[random_index:random_index+M]
                    yi = self.y[random_index:random_index+M]
                    gradient = (1.0/M)*self.gradient_func(self.beta,xi,yi)
                    update_beta = self.tuning_algorithm.update(gradient, iter)
                    self.beta -= update_beta



        else:
            for epoch in range(n_epochs):
                self.tuning_algorithm.reset_params()
                for iter in range(1,m+1):
                    #t = epoch*m + iter
                    #self.tuning_algorithm.eta = self.learning_schedule(5,50,t)
                    random_index = M*random.randint(key,shape=(1,),minval=0,maxval=m)[0]
                    xi = self.X[random_index:random_index+M]
                    yi = self.y[random_index:random_index+M]
                    gradient = (1.0/M)*self.gradient_func(self.beta,xi,yi)
                    mom_term = momentum*jnp.copy(gradient)
                    update_beta = self.tuning_algorithm.update(gradient, iter) + mom_term
                    self.beta -= update_beta


        self.tuning_algorithm.set_tuning_algorithm("DEFAULT")

        return self.beta
