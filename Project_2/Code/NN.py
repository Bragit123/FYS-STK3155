
"""
This code is largely inspired by the lecture notes by Morten Hjort-Jensen at the
following link:
https://compphysics.github.io/MachineLearning/doc/LectureNotes/_build/html/exercisesweek43.html#the-neural-network
"""
import numpy as np
from jax import jacobian, vmap
from sklearn.utils import resample
from copy import copy
from funcs import derivate

class FFNN:
    def __init__(self, dimensions, hidden_act, output_act, cost_func, seed=100, classification = False):
        self.dimensions = dimensions
        self.hidden_act = hidden_act
        self.output_act = output_act
        self.cost_func = cost_func
        self.seed = seed
        self.schedulers_weight = list()
        self.schedulers_bias = list()

        self.weights = list()
        self.a_matrices = list()
        self.z_matrices = list()
        self.classification = classification

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
        if self.classification == True:
            return np.where(res > 0.5, 1, 0)
        else:
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

            if l == len(self.weights) - 1:
                a = self.output_act(z)
            else:
                a = self.hidden_act(z)

            self.z_matrices.append(z)
            self.a_matrices.append(a)

        return a

    def backpropagate(self, X, t, lam):
        cost = self.cost_func(t)
        act_hidden = self.hidden_act
        act_output = self.output_act
        grad_cost = derivate(cost)
        grad_act_hidden = vmap(vmap(derivate(act_hidden)))
        grad_act_output = vmap(vmap(derivate(act_output)))

        for i in range(len(self.weights) - 1, -1, -1):
            # Output layer:
            if i == len(self.weights) - 1:
                dact = grad_act_output(self.z_matrices[i+1])
                dcost = grad_cost(self.a_matrices[i+1])
                delta_matrix = dact * dcost

            # Hidden layers:
            else:
                wdelta = self.weights[i + 1][1:, :] @ delta_matrix.T
                dact = grad_act_hidden(self.z_matrices[i + 1])
                delta_matrix = wdelta.T * dact

            # Calculate gradient
            grad_weights = self.a_matrices[i].T @ delta_matrix
            grad_bias = np.sum(delta_matrix, axis=0).reshape(1, delta_matrix.shape[1])

            # Regularization term
            grad_weights = grad_weights + self.weights[i][1:, :] * lam

            # Use scheduler
            update_matrix = np.vstack(
                [
                    self.schedulers_bias[i].update_change(grad_bias),
                    self.schedulers_weight[i].update_change(grad_weights)
                ]
            )

            # Update weights and bias
            self.weights[i] -= update_matrix

    def train(self, X, t, scheduler, batches=1, epochs=100, lam=0, X_val = None, t_val = None):
        np.random.seed(self.seed)

        # Creating arrays for score metrics
        train_errors = np.empty(epochs)
        train_errors.fill(np.nan)

        train_accs = np.empty(epochs)
        train_accs.fill(np.nan)
        val_set = False
        if X_val is not None:
            val_set = True

        if val_set:
            val_errors = np.empty(epochs)
            val_errors.fill(np.nan)

            val_accs = np.empty(epochs)
            val_accs.fill(np.nan)


        # Create empty lists for schedulers
        self.schedulers_weight = list()
        self.schedulers_bias = list()

        # Compute number of batches
        batch_size = X.shape[0] // batches

        # Resample training data
        X, t = resample(X, t, replace=False)

        # Find cost functions
        cost_func_train = self.cost_func(t)
        if val_set:
            cost_func_val = self.cost_func(t_val)

        # Create schedulers for each weight matrix
        for i in range(len(self.weights)):
            self.schedulers_weight.append(copy(scheduler))
            self.schedulers_bias.append(copy(scheduler))

        print(f"{scheduler.__class__.__name__}: Eta={scheduler.eta}, Lambda={lam}")

        try:
            ## Train the neural network
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
                    self.backpropagate(X_batch, t_batch, lam)

                # Reset schedulers for each epoch
                for scheduler in self.schedulers_weight:
                    scheduler.reset()
                for scheduler in self.schedulers_bias:
                    scheduler.reset()

                # Computing performance metrics
                pred_train = self.predict(X)
                train_error = cost_func_train(pred_train)
                train_errors[e] = train_error

                if val_set:
                    pred_val = self.predict(X_val)
                    val_errors[e] = cost_func_val(pred_val)
                    if self.classification == True:
                        val_accuracy = np.mean(pred_val == t_val)
                        val_accs[e] = val_accuracy

                if self.classification == True:
                    train_accuracy = np.mean(pred_train == t)
                    train_accs[e] = train_accuracy

                progression = e / epochs
                print(f"Progress: {progression*100:.0f}%", end="\r")

        except KeyboardInterrupt:
            ## Allow training to be aborted by keypress
            pass

        # Return performance metrics for the entire run
        scores = dict()

        scores["train_errors"] = train_errors

        if val_set:
            scores["val_errors"] = val_errors

        if self.classification:
            scores["train_accs"] = train_accs

            if val_set:
                scores["val_accs"] = val_accs

        return scores
