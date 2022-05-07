from jax import jit, grad, tree_map
from model import loss
import jax.numpy as jnp

@jit
def GD_update(params, x, y, eta):
    grads = grad(loss)(params, x, y)

    return tree_map(
      lambda p, g: p - eta * g, params, grads
    )


def ada_Grad_update(params, x, y, eta, G):
    grads = grad(loss)(params, x, y)

    G = tree_map(lambda prev, g: prev + g**2, G, grads)

    grads = tree_map(lambda precond, g: g/jnp.sqrt(precond), G, grads)

    return tree_map(
      lambda p, g: p - eta * g, params, grads
    ), G

