
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

def GD(X, y):
    H = 2/n * X.T @ X
    eigvals, eigvecs = np.linalg.eig(H)

    beta = np.random.randn(deg+1, 1)
    gamma = 1/np.max(eigvals)

    n_iter = 10000
    for i in range(n_iter):
        gradient = 2/n * X.T @ (X @ beta - y)
        beta = beta - gamma*gradient
    
    return beta

beta_gd = GD(X, y)

# print(f"beta_ols: {beta_ols}")
# print(f"beta_gd: {beta_gd}")

print(f"beta_difference = {np.linalg.norm(beta_gd-beta_ols)}")