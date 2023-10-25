
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

def learning_schedule(t):
    t0 = 5; t1 = 50
    return t0/(t + t1)

@jit
def cost(beta):
    return 1/n * jnp.linalg.norm(X @ beta - y)**2

dC = grad(cost)

def GD(X, y, n_iter=100):
    # H = 2/n * X.T @ X
    # eigvals, eigvecs = jnp.linalg.eig(H)

    beta = random.normal(key, shape=(deg+1, 1))
    eta = learning_schedule(0)

    for i in range(n_iter):
        gradient = dC(beta)
        beta = beta - eta*gradient
    
    return beta

def GD_mom(X, y, gamma, n_iter=100):
    n, p = jnp.shape(X)

    beta = random.normal(key, shape=(p, 1))
    eta = learning_schedule(0)
    v = 0

    for iter in range(n_iter):
        gradient = dC(beta)
        v = eta*gradient + gamma*v
        beta = beta - v

        t = iter
        eta = learning_schedule(t)
    
    return beta

def SGD(X, y, n_epochs, M):
    n, p = jnp.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = random.normal(key, shape=(p, 1))

    for epoch in range(n_epochs):
        for i in range(m):
            ind = M*random.randint(key, shape=(1,), minval=0, maxval=m)
            ind = ind[0] # ind is an array with one element, som must extract the element.
            xi = X[ind:ind+M, :]
            yi = y[ind:ind+M]
            gradients = dC(beta)
            eta = learning_schedule(epoch*m + i)
            beta = beta - eta*gradients

    return beta

def SGD_mom(X, y, n_epochs, M, gamma):
    n, p = jnp.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = random.normal(key, shape=(p, 1))
    eta = learning_schedule(0)
    v = 0

    for epoch in range(n_epochs):
        for i in range(m):
            ind = M*random.randint(key, shape=(1,), minval=0, maxval=m)
            ind = ind[0] # ind is an array with one element, som must extract the element.
            xi = X[ind:ind+M, :]
            yi = y[ind:ind+M]

            gradient = dC(beta)
            
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
print(f"beta_gd: {jnp.linalg.norm(beta_gd-beta_ols)}")
print(f"beta_mom: {jnp.linalg.norm(beta_mom-beta_ols)}")
print(f"beta_sgd: {jnp.linalg.norm(beta_sgd-beta_ols)}")
print(f"beta_sgd_mom: {jnp.linalg.norm(beta_sgd_mom-beta_ols)}")