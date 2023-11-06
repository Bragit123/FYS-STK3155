
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jacobian
from funcs import sigmoid

# grad_sigmoid = vmap(vmap(grad(sigmoid)))
# grad_sigmoid = jacobian(sigmoid)
# grad_sigmoid = grad(sigmoid)

def f(X):
    return X**2
df = grad(f)

X = np.arange(8, dtype=float).reshape(4,2)
res = np.zeros(8).reshape(4,2)

for i in range(4):
    for j in range(2):
        res[i,j] = df(X[i,j])

print(f"X = {X}")
print(f"Expected = {res}")

df = vmap(vmap(grad(f)))
res = df(X)
print(f"vmap = {res}")

df = vmap(jacobian(f))
res = df(X)
print(f"Jacobian = {res}")