
import jax.numpy as jnp
import numpy as np
from jax import grad, vmap, jacobian
from funcs import sigmoid

# grad_sigmoid = vmap(vmap(grad(sigmoid)))
# grad_sigmoid = jacobian(sigmoid)
# grad_sigmoid = grad(sigmoid)

# def cost(x, t):
#     return -(1.0 / t.size) * jnp.sum(t * jnp.log(x + 10e-10))
def f(x):
    return x**2

df = vmap(vmap(grad(f)))

x = np.arange(10, dtype=float).reshape(5,2)
print(x)
print(df(x))