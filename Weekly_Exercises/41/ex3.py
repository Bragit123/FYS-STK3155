
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(100)
n = 100
x = 2*np.random.rand(n,1)
y = 3*x**2 - 2*x + 7 + 0.1*np.random.randn(n, 1)

deg = 2
poly = PolynomialFeatures(degree=deg)
X = poly.fit_transform(x)

beta_ols = np.linalg.pinv((X.T @ X)) @ X.T @ y

def step_length(t0, t1, t):
    return t0/(t + t1)

def GD(X, y, n_iter=100):
    H = 2/n * X.T @ X
    eigvals, eigvecs = np.linalg.eig(H)

    beta = np.random.randn(deg+1, 1)
    eta = 1/np.max(eigvals)

    for i in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        beta = beta - eta*gradient
    
    return beta

def GD_mom(X, y, gamma, n_iter=100):
    H = 2/n * X.T @ X
    eigvals, eigvecs = np.linalg.eig(H)
    beta = np.random.randn(deg+1, 1)
    eta = 1/np.max(eigvals)
    v = 0

    for iter in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        v = eta*gradient + gamma*v
        beta = beta - v
    
    return beta

def SGD(X, y, n_epochs, batch_size):
    n = len(y)
    H = 2/n * X.T @ X
    eigvals, eigvecs = np.linalg.eig(H)

    M = batch_size # Size of each minibatch
    m = int(n/M) # Number of minibatches
    
    beta = np.random.randn(deg+1, 1)
    t0 = 1
    t1 = 0.01*np.max(eigvals)
    eta = step_length(t0, t1, 0)

    j = 0
    for epoch in range(n_epochs):
        for batch in range(m):
            ind = M * np.random.randint(m)
            Xk = X[ind:ind+M,:]
            yk = y[ind:ind+M,:]
            gradient = 2/n * Xk.T @ (Xk @ beta - yk)
            beta = beta - eta*gradient

            t = epoch*m + batch
            eta = step_length(t0, t1, t)
            j += 1
    
    return beta

def SGD_mom(X, y, n_epochs, batch_size, gamma):
    n = len(y)
    H = 2/n * X.T @ X
    eigvals, eigvecs = np.linalg.eig(H)
    
    M = batch_size
    m = int(n/M)

    beta = np.random.randn(deg+1, 1)
    t0 = 2
    t1 = 0.01*np.max(eigvals)
    eta = step_length(t0, t1, 0)
    v = 0

    for epoch in range(n_epochs):
        for batch in range(m):
            ind = M * np.random.randint(m)
            Xk = X[ind:ind+M,:]
            yk = y[ind:ind+M,:]
            gradient = 2/n * Xk.T @ (Xk @ beta - yk)
            
            v = eta*gradient + gamma*v
            beta = beta - v

            t = epoch*m + batch
            eta = step_length(t0, t1, t)
    
    return beta

n_iter = 1000
beta_gd = GD(X, y, n_iter)

gamma = 0.1
beta_mom = GD_mom(X, y, gamma, n_iter)

batch_size = 5
n_epochs = 10000
beta_sgd = SGD(X, y, n_epochs, batch_size)

beta_sgd_mom = SGD_mom(X, y, n_epochs, batch_size, gamma)

print(f"beta_ols: {beta_ols}")
print(f"beta_gd: {np.linalg.norm(beta_gd-beta_ols)}")
print(f"beta_mom: {np.linalg.norm(beta_mom-beta_ols)}")
print(f"beta_sgd: {np.linalg.norm(beta_sgd-beta_ols)}")
print(f"beta_sgd_mom: {np.linalg.norm(beta_sgd_mom-beta_ols)}")