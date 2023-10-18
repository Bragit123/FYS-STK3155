
import numpy as jnp
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

def GD(X, y, n_iter=100):
    H = 2/n * X.T @ X
    eigvals, eigvecs = jnp.linalg.eig(H)

    beta = random.normal(key, shape=(deg+1, 1))
    gamma = 1/jnp.max(eigvals)

    for i in range(n_iter):
        gradient = dC(beta)
        beta = beta - gamma*gradient
    
    return beta

def GD_mom(X, y, gamma, n_iter=100):
    H = 2/n * X.T @ X
    eigvals, eigvecs = jnp.linalg.eig(H)
    beta = random.normal(key, shape=(deg+1, 1))
    eta = 1/jnp.max(eigvals)
    gamma = 0.1
    v = 0

    for iter in range(n_iter):
        gradient = dC(beta)
        v = eta*gradient + gamma*v
        beta = beta - v
    
    return beta

n_iter = 1000
beta_gd = GD(X, y, n_iter)

gamma = 0.1
beta_mom = GD_mom(X, y, gamma, n_iter)

# print(f"beta_ols: {beta_ols}")
# print(f"beta_gd: {beta_gd}")
# print(f"beta_mom: {beta_mom}")

print("beta_difference")
print(f"beta_gd: {jnp.linalg.norm(beta_gd-beta_ols)}")
print(f"beta_mom: {jnp.linalg.norm(beta_mom-beta_ols)}")