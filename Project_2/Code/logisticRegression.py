import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from jax import grad as jax_grad
from typing import Callable

# Load the data
np.random.seed(2)
cancer = load_breast_cancer()

import scheduler
from scheduler import Adagrad
from funcs import *


class LogisticRegression: # target and y  is the same, and beta is my weigths  ?
    

    def __init__(self, beta: np.array, X: np.ndarray, Target: np.ndarray, costfunction: callable , scheduler_class: type,  n: int = 100): 
        """
    Initialize a class instance for your custom logistic regression model.

    Parameters:
    - beta (np.array): Initial parameter values for the model.
    - X (np.ndarray): Input features for training the model.
    - Target (np.ndarray): Target values for training the model.
    - costfunction (callable, optional): Cost function used for optimization. Default is CostCrossEntropy.
    - Scheduler (scheduler.Scheduler, optional): Learning rate scheduler. Default is None.
    - n (int, optional): Number of iterations for optimization. Default is 100.

    """
        self.X = X
        self.Target = Target
        self.n = n
        self.Scheduler = scheduler_class
        self.Cost_func = costfunction
        self.beta  = beta
        #print(self.Scheduler)
    

    def learning_schedule(self, t):
        t0 = 5; t1 = 50
        return t0/(t + t1)    
    

    def SGD(self, n_epochs = 10):

        n = 100
        gamma  =  0.01  

        M = 10  #size of each minibatch
        m = int(n/M) #number of minibatches

        eta = self.learning_schedule(0)

        scheduler_SGD = self.Scheduler(eta)
        #print(scheduler_SGD)
        #cost_func = CostCrossEntropy(self.Target)

        beta_start  = self.beta
        

        for epoch in range(1, n_epochs + 1000): #bytter ut beta med weigths
            for i in range(m):
                k = np.random.randint(m) 

                #self.X[k:k+m]      #lagre 
                #self.Target[k:k+m]
                X = self.X[k:k+m, :]      #lagre 
                Target = self.Target[k:k+m, :]

                cost_func = self.Cost_func(Target)  #target is beta 

                derivative_Cost_func = jax_grad(cost_func)

                gradient = derivative_Cost_func(X)

                v = scheduler_SGD.update_change(gradient)

                self.beta = self.beta - v
            #print(f"self.beta: ")
            #print(beta_start - self.beta)
            #print("v")
            #print(v)

    
        return self.beta
    


    def predict(self):

        print(f"beta.shape: {self.beta.shape}")
        print(f"X.shape: {self.X.shape}")

        probabilities = 1 / (1 + np.exp(- self.X @ self.beta ))
        print("probabilities")
        print(probabilities)
        print("beta")
        #print(self.beta)
        print("X")
        #print(X)

        #converting predictions to binary output

        predictions = (probabilities >= 0.5).astype(int) #Converting to 0 or 1s, and then compare to the target gate
        print("predictions")
        print(predictions)
        print("self.Target")
        print(self.Target)

        accuracy = np.mean(predictions == self.Target)
        print("accuracy")
        print(accuracy)

        return accuracy


    def __str__(self):
        return f"Accuracy of model SGD: {self.predict()}"



    def Scikit_SGD(self, X, y, solver1):

        #X_train, X_test, y_train, y_test = train_test_split(dataset.dataset, dataset.target, random_state=0)

        logreg = LogisticRegression(solver = solver1) #code taken from LG slide 
        #logreg.fit(X_train, y_train) 

        scaler = StandardScaler()
        scaler.fit(X)

        X_scaled = scaler.transform(X)
        #X_test_scaled = scaler.transform(X_test)

        logreg.fit(X, y)

        return logreg.score(X,y)
    

    def __str__(self):
        return f"Accuracy of model SciKit: {self.Scikit_SGD()}"



    
#X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)


#X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)

#yXOR = np.c_[[0,1,1,1]]

X = cancer.data
target = cancer.target 
print(f"X.shape {X.shape}")
print(f"target: {target.shape}")
beta = np.random.randn(X.shape[1],1)

instance = LogisticRegression
           

#target = yXOR

#a = target * jnp.log(X + 10e-10)

#cost_funtion = CostCrossEntropy(yXOR)

obj = instance(beta, X, target, CostCrossEntropy, Adagrad, 100)



print(obj.predict())
