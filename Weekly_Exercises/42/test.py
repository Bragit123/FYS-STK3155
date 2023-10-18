
import jax.numpy as jnp
from jax import random

key = random.PRNGKey(100)

n = 100; M = 5
x = jnp.arange(n)
m = int(n/M)
a = M*random.randint(key, shape=(1,), minval=0, maxval=m)
a = a[0]
xi = x[a:a+M]

print(f"a = {a}")
print(f"xi = {xi}")