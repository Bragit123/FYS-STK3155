
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import jax.numpy as jnp
from jax import grad, random, jit

key = random.PRNGKey(500)
n = 100
x = 2 * random.uniform(key, shape=(n,1))
y = 4 + 3*x + random.normal(key, shape=(n,1))

deg = 1
poly = PolynomialFeatures(degree=deg)
X = poly.fit_transform(x)

beta_ols = jnp.linalg.pinv((X.T @ X)) @ X.T @ y

@jit
def cost(beta):
    return 1/n * jnp.linalg.norm(X @ beta - y)**2

dC = grad(cost)

def GD(X, y):
    H = 2/n * X.T @ X
    eigvals, eigvecs = jnp.linalg.eig(H)

    beta = random.normal(key, shape=(deg+1, 1))
    gamma = 1/jnp.max(eigvals)

    n_iter = 1000

    for i in range(n_iter):
        gradient = dC(beta)
        beta = beta - gamma*gradient
    
    return beta

beta_gd = GD(X, y)

print(f"beta_difference = {jnp.linalg.norm(beta_gd-beta_ols)}")