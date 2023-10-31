
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(100)
n = 100
x = 2*np.random.rand(n,1)
y = 2 + 3*x + 4*x**2# + np.random.randn(n,1)


deg = 2
poly = PolynomialFeatures(degree=deg)
X = poly.fit_transform(x)

beta_ols = np.linalg.pinv((X.T @ X)) @ X.T @ y

def learning_schedule(t):
    t0 = 5; t1 = 50
    return t0/(t + t1)

def GD_adam(X, y, n_iter=100):
    n, p = np.shape(X)
    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)

    beta1 = 0.9
    beta2 = 0.999
    delta = 1e-7 # Value recommended by Goodfellow et. al

    iter = 0
    first_moment = 0.0
    second_moment = 0.0
    for i in range(n_iter):
        iter += 1
        gradient = 2/n * X.T @ (X @ beta - y)

        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient**2

        first_term = first_moment / (1.0 - beta1**iter)
        second_term = second_moment / (1.0 - beta2**iter)
        
        update = eta * first_term / (np.sqrt(second_term) + delta)
        beta -= update
    
    return beta

def GD_mom_adam(X, y, gamma, n_iter=100):
    n, p = np.shape(X)

    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)
    v = 0

    beta1 = 0.9
    beta2 = 0.999
    delta = 1e-7 # Value recommended by Goodfellow et. al

    iter = 0
    first_moment = 0.0
    second_moment = 0.0
    for i in range(n_iter):
        iter += 1
        gradient = 2/n * X.T @ (X @ beta - y)

        first_moment = beta1 * first_moment + (1 - beta1) * gradient
        second_moment = beta2 * second_moment + (1 - beta2) * gradient**2

        first_term = first_moment / (1.0 - beta1**iter)
        second_term = second_moment / (1.0 - beta2**iter)

        update = eta * first_term / (np.sqrt(second_term) + delta)
        v = update + gamma*v
        beta -= v

        t = i
        eta = learning_schedule(t)
    
    return beta

def SGD_adam(X, y, n_epochs, M):
    n, p = np.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = np.random.randn(p,1)
    eta = learning_schedule(0)

    beta1 = 0.9
    beta2 = 0.999
    delta = 1e-7 # Value recommended by Goodfellow et. al

    iter = 0
    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        iter += 1
        for i in range(m):
            ind = M*np.random.randint(m)
            xi = X[ind:ind+M]
            yi = y[ind:ind+M]
            gradient = (2.0/M) * xi.T @ ((xi @ beta) - yi)

            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * gradient**2

            first_term = first_moment / (1.0 - beta1**iter)
            second_term = second_moment / (1.0 - beta2**iter)
            
            update = eta * first_term / (np.sqrt(second_term) + delta)
            beta -= update

            eta = learning_schedule(epoch*m + i)

    return beta

def SGD_mom_adam(X, y, n_epochs, M, gamma):
    n, p = np.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)
    v = 0

    beta1 = 0.9
    beta2 = 0.999
    delta = 1e-7 # Value recommended by Goodfellow et. al

    iter = 0
    for epoch in range(n_epochs):
        first_moment = 0.0
        second_moment = 0.0
        iter += 1
        for i in range(m):
            ind = M * np.random.randint(m)
            xi = X[ind:ind+M, :]
            yi = y[ind:ind+M]

            gradient = 2.0/M * xi.T @ (xi @ beta - yi)

            first_moment = beta1 * first_moment + (1 - beta1) * gradient
            second_moment = beta2 * second_moment + (1 - beta2) * gradient**2

            first_term = first_moment / (1.0 - beta1**iter)
            second_term = second_moment / (1.0 - beta2**iter)

            update = eta * first_term / (np.sqrt(second_term) + delta)
            v = update + gamma*v
            beta -= v

            t = epoch*m + i
            eta = learning_schedule(t)

    return beta

n_iter = 1000
beta_gd = GD_adam(X, y, n_iter)

gamma = 0.1
beta_mom = GD_mom_adam(X, y, gamma, n_iter)

M = 5
n_epochs = 50
beta_sgd = SGD_adam(X, y, n_epochs, M)

beta_sgd_mom = SGD_mom_adam(X, y, n_epochs, M, gamma)

print("beta_difference (with adam)")
print(f"beta_gd: {np.linalg.norm(beta_gd-beta_ols)}")
print(f"beta_mom: {np.linalg.norm(beta_mom-beta_ols)}")
print(f"beta_sgd: {np.linalg.norm(beta_sgd-beta_ols)}")
print(f"beta_sgd_mom: {np.linalg.norm(beta_sgd_mom-beta_ols)}")