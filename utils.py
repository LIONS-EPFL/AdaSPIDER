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
        bound = 0.1/np.sqrt(n_in)
        params.append(
            dict(
                weights= (np.random.uniform(-3.0*bound, 3.0*bound, size=(n_in, n_out))),
                biases= (np.random.uniform(-bound, bound, size=(n_out,))),
            )
        )
    return params


@partial(jit, static_argnames=["k", "dtype"])
def one_hot(x, k, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def compute_gradient_norm(params, net_state, x, y):
    s = 0
    for i in range(0, 1875):
        images, labels = x[32*i: 32*(i+1)], y[32*i: 32*(i+1)]
        grads, _ = grad(loss, has_aux=True)(params, net_state, images, labels, False)
        norms = jax.tree_map(lambda v: jnp.sum(v * v), grads)
        s += 32*jax.tree_util.tree_reduce(lambda a, b: a + b, norms, 0.0)
    return s/60000

def compute_distance(params, starting_params):
  distances = tree_map(lambda a, b: jnp.linalg.norm(a - b)**2, params, starting_params)
  return jax.tree_util.tree_reduce(lambda a, b: a+b, distances, 0.0)