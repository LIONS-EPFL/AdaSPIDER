import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, grad, tree_map
from jax.tree_util import tree_reduce
from functools import partial
from model import loss


def create_params(layer_widths):
    params = []
    for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
        params.append(
            dict(
                weights=0.0001 * jnp.array(np.random.normal(size=(n_in, n_out))),
                biases=0.0001 * jnp.array(np.random.normal(size=(n_out,))),
            )
        )
    return params


@partial(jit, static_argnames=["k", "dtype"])
def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

@jit
def compute_gradient_norm(params, x, y):
    grads = grad(loss)(params, x, y)
    norms = jax.tree_map(lambda v: jnp.sum(v * v), grads)
    return jax.tree_util.tree_reduce(lambda a, b: a + b, norms, 0.0)

def compute_distance(params, starting_params):
  distances = tree_map(lambda a, b: jnp.linalg.norm(a - b)**2, params, starting_params)
  return jax.tree_util.tree_reduce(lambda a, b: a+b, distances, 0.0)