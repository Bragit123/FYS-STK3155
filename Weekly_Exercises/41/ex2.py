
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

def GD(X, y, n_iter=100):
    H = 2/n * X.T @ X
    eigvals, eigvecs = np.linalg.eig(H)

    beta = np.random.randn(deg+1, 1)
    gamma = 1/np.max(eigvals)

    for i in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        beta = beta - gamma*gradient
    
    return beta

def GD_mom(X, y, gamma, n_iter=100):
    H = 2/n * X.T @ X
    eigvals, eigvecs = np.linalg.eig(H)
    beta = np.random.randn(deg+1, 1)
    eta = 1/np.max(eigvals)
    gamma = 0.1
    v = 0

    for iter in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        v = eta*gradient + gamma*v
        beta = beta - v
    
    return beta

n_iter = 1000
beta_gd = GD(X, y, n_iter)

gamma = 0.1
beta_mom = GD_mom(X, y, gamma, n_iter)

print(f"beta_ols: {beta_ols}")
print(f"beta_gd: {beta_gd}")
print(f"beta_mom: {beta_mom}")