import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, grad
from functools import partial
from model import loss

def create_params(layer_widths):
  params = []
  for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
    params.append(
        dict(weights=0.01*np.random.normal(size=(n_in, n_out)),
             biases=0.01*np.random.normal(size=(n_out,))
            )
    )
  return params

@partial(jit, static_argnames=['k', 'dtype'])
def one_hot(x, k, dtype=jnp.float32):
  return jnp.array(x[:, None] == jnp.arange(k), dtype)

def create_state(layer_widths):
  state = []
  for n_in, n_out in zip(layer_widths[:-1], layer_widths[1:]):
    state.append(
        dict(weights=np.zeros((n_in, n_out)) + 1e-3,
             biases=np.zeros(n_out) + 1e-3
            )
    )
  return state

def compute_gradient_norm(params, x, y):
  grads = grad(loss)(params, x, y)
  norms = jax.tree_map(lambda v: jnp.sum(v*v), grads)
  return jax.tree_util.tree_reduce(lambda a, b: a + b, norms, 0.0)