import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from sklearn.model_selection import  train_test_split 
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from jax import grad as jax_grad


class LogisticRegresssion: # target and y  is the same, and beta is my weigths  ?

    def __init__(self, Name: str, X: np.ndarray, Target: np.ndarray, n: int = 100) -> None: 
        """
    Initialize a class instance for your custom logistic regression model.

    Parameters:
    Name (str): A descriptive name for your model.
    X (np.ndarray): The feature matrix, where each row represents a data sample, and each column represents a feature.
    Target (np.ndarray): The target labels for multiclass classification.
    n (int, optional): The number of iterations for model training. Default is 100.

    Returns:
    None
    """
        self.Name = Name
        self.X = X
        self.Target = Target
        self.n = n

    def learning_schedule(t):
        t0 = 5; t1 = 50
        return t0/(t + t1)
    
    
    def CostFunctionTargets(self, target):

        def cost_func(X):  

            CrossEntropy =  -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

            return  CrossEntropy
        
        

        
        return cost_func
          
       
                
    
          
    def Algorithms_for_learning_rate(self): #y er byttet ut med self.Target

        
        #Learning rates without momentum

        if self.Name == "ADAGRAD":
            delta = 1e-7
            eta = self.learning_schedule(0)
            gradient = 2/self.n * self.X.T @ (self.X @ self.beta - self.Target) #beta is my weigths 
            self.Giter += gradient * gradient   
            eta_ada = eta / (delta + np.sqrt(self.Giter))

            return eta_ada, gradient
        
        if self.Name == "Adagrad_JAX":
            delta = 1e-7
            eta = self.learning_schedule(0) 
            derivative_Cost_func = jax_grad(self.cost_function) #create cost-func method, 
            gradient = derivative_Cost_func(beta, self.X, self.Target)
            self.Giter += gradient * gradient   
            eta_ada_jax = eta / (delta + np.sqrt(self.Giter))

            return eta_ada_jax, gradient

        
        if self.Name == "RMSprop":
            eta = self.learning_schedule(0)
            rho = 0.99
            delta = 1e-7

            gradient = 2/self.n * self.X.T @ (self.X @ self.beta - self.Target)

            self.Giter = (rho * self.Giter + (1 - rho) * gradient * gradient)

            eta_RMSprop = eta / (delta + np.sqrt(self.Giter))

            return eta_RMSprop, gradient
        
        if self.Name == "RMSprop_JAX":
            delta = 1e-7
            eta = self.learning_schedule(0) #is it correct to set t=0 and not evolve it without momentum ? 
            derivative_Cost_func = jax_grad(self.Cost_function)
            gradient = derivative_Cost_func(beta, self.X, self.y)
            self.Giter +=  (rho * self.Giter + (1 - rho) * gradient * gradient)   
            eta_ada_jax = eta / (delta + np.sqrt(self.Giter)) #eta_ada_jax is the learning rate

            return eta_ada_jax, gradient


        if self.Name == "ADAM":               # her er det en feil fra før av vet jeg  
            # Value for learning rate
            eta = self.learning_schedule(0)
            # Value for parameters beta1 and beta2, see https://arxiv.org/abs/1412.6980
            beta1 = 0.9
            beta2 = 0.999
            # Including AdaGrad parameter to avoid possible division by zero
            delta  = 1e-7
            iter = 0

            gradient = 2/self.n * self.X.T @ (self.X @ beta - self.y)

            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * gradient * gradient
            first_term = first_moment / (1.0 - beta1**iter)
            second_term = second_moment/(1.0 - beta2**iter)
            # Scaling with rho the new and the previous results
            eta_Adam = eta * first_term / (np.sqrt(second_term) + delta)

            return eta_Adam, gradient

    def SGD(self, n_epochs = 10):
        n = 100
        # Hessian matrix
        H = (2.0/n)* X.T @ X

        # Get the eigenvalues
        EigValues, EigVectors = np.linalg.eig(H)

        gamma = 1 / np.max(EigValues)

        beta = np.random.randn(2,1)
        
        M = 10  #size of each minibatch
        m = int(n/M) #number of minibatches
        
        for epoch in range(1, n_epochs + 1):
            for i in range(m):
                k = np.random.randint(m) 
                self.X[k:k+m, :], 
                self.Target[k:k+m, :]
                gradient, eta_ada_jax  = self.Algorithms_for_learning_rate("Adagrad_JAX", self.X, self.Target, beta) #bare bruker adagrad_JAX per no,
                #gradient = (2.0/n) * self.X[k:k + m,:].T @ (self.X[k:k + m,:] @ beta - self.y[k:k + m,:])

                beta = beta - eta_ada_jax * gradient

                beta -= gamma*gradient 

        return beta
    

    def Scikit_SGD(self,dataset):

        X_train, X_test, y_train, y_test = train_test_split(dataset.dataset, dataset.target, random_state=0)

        logreg = LogisticRegression(solver='lbfgs') #code taken from LG slide 
        #logreg.fit(X_train, y_train) 

        scaler = StandardScaler()
        scaler.fit(X_train)

        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        logreg.fit(X_train_scaled, y_train)

        return logreg.score(X_test_scaled,y_test)

        







