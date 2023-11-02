
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap

def f(x):
    return jnp.exp(x)

# df = vmap(grad(f))
df = vmap(grad(f), out_axes=1)

# X = np.array([[1,2],[3,4],[5,6]], dtype=float)
X = np.array([[1],[2],[3]], dtype=float)
# X = np.array([1,2,3], dtype=float)
# print(df(X[:,0]))
print(df(X))