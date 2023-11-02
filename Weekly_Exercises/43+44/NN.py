
import numpy as np
from jax import grad, jacobian, vmap
from sklearn.utils import resample
from copy import copy

class FFNN:
    def __init__(self, dimensions, act_func, cost_func, seed=100):
        self.dimensions = dimensions
        self.act_func = act_func
        self.cost_func = cost_func
        self.seed = seed
        self.schedulers_weight = list()
        self.schedulers_bias = list()
        
        self.weights = list()
        self.a_matrices = list()
        self.z_matrices = list()

        self.reset_weights()
    
    def reset_weights(self):
        np.random.seed(self.seed)
        n_layers = len(self.dimensions)
        self.weights = list()

        for i in range(n_layers - 1):
            weight_shape = (self.dimensions[i] + 1, self.dimensions[i + 1])
            weight_arr = np.random.normal(size=weight_shape)
            weight_arr[0,:] = np.random.normal(size=self.dimensions[i + 1]) * 0.01 # Bias
            self.weights.append(weight_arr)
    
    def predict(self, X):
        res = self.feedforward(X)
        return res

    def feedforward(self, X):
        # Reset matrices
        self.a_matrices = list()
        self.z_matrices = list()

        a = X
        self.a_matrices.append(a)
        self.z_matrices.append(a)

        for l in range(len(self.weights)):
            w_mat = self.weights[l]
            z = a @ w_mat[1:,:] + w_mat[0,:]
            a = self.act_func(z)

            self.z_matrices.append(z)
            self.a_matrices.append(a)
        
        return a
    
    def backpropagate(self, X, t, lam):
        cost = self.cost_func(t)
        act = self.act_func
        grad_cost = grad(cost)
        grad_act = vmap(grad(act))

        for i in range(len(self.weights) - 1, -1, -1):
            # Output layer:
            if i == len(self.weights) - 1:
                dact = grad_act(self.z_matrices[i+1].ravel())
                dcost = grad_cost(self.a_matrices[i+1])
                delta_matrix = dact * dcost
            # Hidden layers:
            else:
                wdelta = self.weights[i + 1][1:, :] @ delta_matrix.T
                dact = grad_act(self.z_matrices[i + 1])
                delta_matrix = wdelta.T * dact
            
            # Calculate gradient
            grad_weights = self.a_matrices[i][:, 1:].T @ delta_matrix
            grad_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

            # Regularization term
            grad_weights += self.weights[i][1:, :] * lam

            # Use scheduler
            print(f"grad_b = {grad_bias}")
            print(f"sched_b = {self.schedulers_bias[i].update_change(grad_bias)}")
            print(f"grad_w = {grad_weights}")
            print(f"sched_w = {self.schedulers_weight[i].update_change(grad_weights)}")
            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(grad_bias),
                    self.schedulers_weight[i].update_change(grad_weights)
                ]
            )

            # Update weights and bias
            print(self.weights[i])
            print(update_matrix)
            print(i)
            self.weights[i] -= update_matrix 
    
    def train(self, X, t, scheduler, batches=1, epochs=100, lam=0):
        np.random.seed(self.seed)

        # Creating arrays for score metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)

        self.schedulers_weight = list()
        self.schedulers_bias = list()

        batch_size = X.shape[0] // batches

        X, t = resample(X, t)
        cost_func_train = self.cost_func(t)

        # Create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))
        
        print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lam}")

        try:
            for e in range(epochs):
                for i in range(batches):
                    if i == batches - 1:
                        # If this is the last batch, take all that's left.
                        X_batch = X[i * batch_size :, :]
                        t_batch = t[i * batch_size :, :]
                    else:
                        X_batch = X[i * batch_size : (i + 1) * batch_size, :]
                        t_batch = t[i * batch_size : (i + 1) * batch_size, :]
                    
                    self.feedforward(X_batch)
                    print()
                    print(f"{e}.{i}:")
                    print(f"z_mat = {self.z_matrices}")
                    print(f"a_mat = {self.a_matrices}")
                    self.backpropagate(X_batch, t_batch, lam)
                    # self.backback(X_batch, t_batch, lam)
                
                # Reset schedulers for each epoch
                for scheduler in self.schedulers_weight:
                    scheduler.reset()
                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # Computing performance metrics
                pred_train = self.predict(X)
                train_error = cost_func_train(pred_train)

                train_errors[e] = train_error

                progression = e / epochs
                print(f"Progress: {progression*100}%")

        except KeyboardInterrupt:
            pass

        # Return performance metrics for the entire run
        scores = dict()
        scores["train_errors"] = train_errors

        return