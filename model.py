from optax import softmax_cross_entropy
from jax import jit
from jax.nn import relu, elu
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@jit
def net(params, x):
    *hidden, last = params
    for layer in hidden:
        x = elu(x @ layer["weights"] + layer["biases"])
    return x @ last["weights"] + last["biases"]


@jit
def loss(params, x, y):
    out = net(params, x)
    logits = out - logsumexp(out, axis=1).reshape(-1, 1)
    return -jnp.mean(y * logits)


@jit
def accuracy(params, x, y):
    true_class = jnp.argmax(y, axis=1)
    predicted_class = jnp.argmax(net(params, x), axis=1)
    return jnp.mean(predicted_class == true_class)
