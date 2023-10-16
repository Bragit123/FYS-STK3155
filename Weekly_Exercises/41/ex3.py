
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(100)
n = 100
x = 2*np.random.rand(n,1)
y = 4 + 3*x + np.random.randn(n,1)


deg = 1
poly = PolynomialFeatures(degree=deg)
X = poly.fit_transform(x)

beta_ols = np.linalg.pinv((X.T @ X)) @ X.T @ y

def learning_schedule(t):
    t0 = 5; t1 = 50
    return t0/(t + t1)

def GD(X, y, n_iter=100):
    # H = 2/n * X.T @ X
    # eigvals, eigvecs = np.linalg.eig(H)

    beta = np.random.randn(deg+1, 1)
    eta = learning_schedule(0)

    for i in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        beta = beta - eta*gradient
    
    return beta

def GD_mom(X, y, gamma, n_iter=100):
    n, p = np.shape(X)

    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)
    v = 0

    for iter in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        v = eta*gradient + gamma*v
        beta = beta - v

        t = iter
        eta = learning_schedule(t)
    
    return beta

def SGD(X, y, n_epochs, M):
    n, p = np.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = np.random.randn(p,1)

    for epoch in range(n_epochs):
        for i in range(m):
            ind = M*np.random.randint(m)
            xi = X[ind:ind+M, :]
            yi = y[ind:ind+M]
            gradients = (2.0/M) * xi.T @ ((xi @ beta) - yi)
            eta = learning_schedule(epoch*m + i)
            beta = beta - eta*gradients

    return beta

def SGD_mom(X, y, n_epochs, M, gamma):
    n, p = np.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)
    v = 0

    for epoch in range(n_epochs):
        for i in range(m):
            ind = M * np.random.randint(m)
            xi = X[ind:ind+M, :]
            yi = y[ind:ind+M]

            gradient = 2.0/M * xi.T @ (xi @ beta - yi)
            
            v = eta*gradient + gamma*v
            beta = beta - v

            t = epoch*m + i
            eta = learning_schedule(t)

    return beta

n_iter = 1000
beta_gd = GD(X, y, n_iter)

gamma = 0.1
beta_mom = GD_mom(X, y, gamma, n_iter)

M = 5
n_epochs = 50
beta_sgd = SGD(X, y, n_epochs, M)

beta_sgd_mom = SGD_mom(X, y, n_epochs, M, gamma)

print("beta_difference")
print(f"beta_gd: {np.linalg.norm(beta_gd-beta_ols)}")
print(f"beta_mom: {np.linalg.norm(beta_mom-beta_ols)}")
print(f"beta_sgd: {np.linalg.norm(beta_sgd-beta_ols)}")
print(f"beta_sgd_mom: {np.linalg.norm(beta_sgd_mom-beta_ols)}")