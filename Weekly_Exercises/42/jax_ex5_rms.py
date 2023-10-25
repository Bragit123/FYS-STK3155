
from sklearn.preprocessing import PolynomialFeatures
import jax.numpy as jnp
from jax import grad, random, jit

key = random.PRNGKey(100)
n = 100
x = 2 * random.uniform(key, shape=(n,1))
y = 4 + 3*x + random.normal(key, shape=(n,1))


deg = 2
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

def GD_rms(X, y, n_iter=100):
    n, p = jnp.shape(X)
    beta = random.normal(key, shape=(p, 1))
    eta = learning_schedule(0)

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate

    Giter = 0 # Initialize gradient accumulation variable

    for i in range(n_iter):
        gradient = dC(beta)
        
        Giter = rho * Giter + (1 - rho) * gradient**2
        beta = beta - eta/(delta + jnp.sqrt(Giter)) * gradient
    
    return beta

def GD_mom_rms(X, y, gamma, n_iter=100):
    n, p = jnp.shape(X)

    beta = random.normal(key, shape=(p, 1))
    eta = learning_schedule(0)
    v = 0

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate
    Giter = 0 # Initialize gradient accumulation variable

    for iter in range(n_iter):
        gradient = dC(beta)

        Giter = rho * Giter + (1 - rho) * gradient**2
        eta_rms = eta/(delta + jnp.sqrt(Giter))
        v = eta_rms*gradient + gamma*v
        beta = beta - v

        t = iter
        eta = learning_schedule(t)
    
    return beta

def SGD_rms(X, y, n_epochs, M):
    n, p = jnp.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = random.normal(key, shape=(p, 1))
    eta = learning_schedule(0)

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate

    for epoch in range(n_epochs):
        Giter = 0 # Initialize gradient accumulation variable
        for i in range(m):
            ind = M*random.randint(key, shape=(1,), minval=0, maxval=m)
            ind = ind[0] # ind is an array with one element, som must extract the element.
            xi = X[ind:ind+M]
            yi = y[ind:ind+M]
            gradient = dC(beta)

            Giter = rho * Giter + (1 - rho) * gradient**2
            eta_rms = eta/(delta + jnp.sqrt(Giter))
            beta = beta - eta_rms*gradient

            eta = learning_schedule(epoch*m + i)

    return beta

def SGD_mom_rms(X, y, n_epochs, M, gamma):
    n, p = jnp.shape(X) # p = degree + 1
    m = int(n/M) # M = batch_size ; m = number of minibatches

    beta = random.normal(key, shape=(p, 1))
    eta = learning_schedule(0)
    v = 0

    delta = 1e-7 # Value recommended by Goodfellow et. al
    rho = 1e-8 # Decay rate

    for epoch in range(n_epochs):
        Giter = 0 # Initialize gradient accumulation variable
        for i in range(m):
            ind = M*random.randint(key, shape=(1,), minval=0, maxval=m)
            ind = ind[0] # ind is an array with one element, som must extract the element.
            xi = X[ind:ind+M, :]
            yi = y[ind:ind+M]

            gradient = dC(beta)

            Giter = rho * Giter + (1 - rho) * gradient**2
            eta_rms = eta/(delta + jnp.sqrt(Giter))
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
print(f"beta_gd: {jnp.linalg.norm(beta_gd-beta_ols)}")
print(f"beta_mom: {jnp.linalg.norm(beta_mom-beta_ols)}")
print(f"beta_sgd: {jnp.linalg.norm(beta_sgd-beta_ols)}")
print(f"beta_sgd_mom: {jnp.linalg.norm(beta_sgd_mom-beta_ols)}")