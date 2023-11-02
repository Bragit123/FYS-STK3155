
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap

def f(x):
    return jnp.exp(x)

df = vmap(grad(f))

a = np.array([1,2,3], dtype=float)
A = np.array([[1],[2],[3]], dtype=float)

print(a, A)

aa = df(a)
print(aa)

# A = np.reshape(A, (A.shape[0],1))
A = A.ravel()
print(A)
AA = df(A)
print(AA)