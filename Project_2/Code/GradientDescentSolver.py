
import numpy as np
from typing import Callable
import jax

class Gradient_Descent:
    def __init__(self, X, y, cost_func):
        self.X = X
        self.y = y
        self.cost_func = cost_func
        self.gradient_func = None
        self.tuning_algorithms = ["DEFAULT", "ADAGRAD", "RMSPROP", "ADAM"] # List of available tuning algorithms in CAPS
        self.tuning = 0 # Index of 'active' tuning algorithm (chosen from tuning_algorithms)
    
    def set_tuning_algorithm(self, algorithm: str):
        """ Choose algorithm for tuning the learning rate """
        algorithm = algorithm.upper()
        
        algos = self.tuning_algorithms
        if algorithm in algos:
            self.tuning = algos.index(algorithm)
        else:
            self.tuning = 0
            print("WARNING: Invalid tuning algorithm. Changed back to default (no tuning).")
    
    def learning_schedule(t0, t1, t):
        """ Compute learning rate (might want to have t0 and t1 as class
        attributes instead) """
        return t0/(t + t1)

    def GD(self, n_iter, momentum = None, use_jax = True, tuning_algorithm = "adagrad"):
        """ Compute gradient descent (with momentum if specified)"""
        return 0
    
    def SGD(self, n_epochs, batch_size, momentum = None, use_jax = True):
        """ Compute stochastic gradient descent (with momentum if specified)"""
        return 0