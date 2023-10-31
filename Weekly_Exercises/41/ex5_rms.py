
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

def GD_rms(X, y, n_iter=100):
    n, p = np.shape(X)
    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate

    Giter = 0 # Initialize gradient accumulation variable

    for i in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        
        Giter = rho * Giter + (1 - rho) * gradient**2
        beta = beta - eta/(delta + np.sqrt(Giter)) * gradient
    
    return beta

def GD_mom_rms(X, y, gamma, n_iter=100):
    n, p = np.shape(X)

    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)
    v = 0

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate
    Giter = 0 # Initialize gradient accumulation variable

    for iter in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)

        Giter = rho * Giter + (1 - rho) * gradient**2
        eta_rms = eta/(delta + np.sqrt(Giter))
        v = eta_rms*gradient + gamma*v
        beta = beta - v

        t = iter
        eta = learning_schedule(t)
    
    return beta

def SGD_rms(X, y, n_epochs, M):
    n, p = np.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = np.random.randn(p,1)
    eta = learning_schedule(0)

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate

    for epoch in range(n_epochs):
        Giter = 0 # Initialize gradient accumulation variable
        for i in range(m):
            ind = M*np.random.randint(m)
            xi = X[ind:ind+M]
            yi = y[ind:ind+M]
            gradient = (2.0/M) * xi.T @ ((xi @ beta) - yi)

            Giter = rho * Giter + (1 - rho) * gradient**2
            eta_rms = eta/(delta + np.sqrt(Giter))
            beta = beta - eta_rms*gradient

            eta = learning_schedule(epoch*m + i)

    return beta

def SGD_mom_rms(X, y, n_epochs, M, gamma):
    n, p = np.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = np.random.randn(p, 1)
    eta = learning_schedule(0)
    v = 0

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate

    for epoch in range(n_epochs):
        Giter = 0 # Initialize gradient accumulation variable
        for i in range(m):
            ind = M * np.random.randint(m)
            xi = X[ind:ind+M, :]
            yi = y[ind:ind+M]

            gradient = 2.0/M * xi.T @ (xi @ beta - yi)

            Giter = rho * Giter + (1 - rho) * gradient**2
            eta_rms = eta/(delta + np.sqrt(Giter))
            v = eta_rms*gradient + gamma*v
            beta = beta - v

            t = epoch*m + i
            eta = learning_schedule(t)

    return beta

n_iter = 1000
beta_gd = GD_rms(X, y, n_iter)

gamma = 0.1
beta_mom = GD_mom_rms(X, y, gamma, n_iter)

M = 5
n_epochs = 50
beta_sgd = SGD_rms(X, y, n_epochs, M)

beta_sgd_mom = SGD_mom_rms(X, y, n_epochs, M, gamma)

print("beta_difference (with RMSprop)")
print(f"beta_gd: {np.linalg.norm(beta_gd-beta_ols)}")
print(f"beta_mom: {np.linalg.norm(beta_mom-beta_ols)}")
print(f"beta_sgd: {np.linalg.norm(beta_sgd-beta_ols)}")
print(f"beta_sgd_mom: {np.linalg.norm(beta_sgd_mom-beta_ols)}")