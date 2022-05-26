from optax import softmax_cross_entropy
from jax import jit
from jax.nn import relu, elu
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import haiku as hk
from functools import partial



def net(x, is_training=True):
    model = hk.nets.ResNet18(num_classes=10)
    return model(x, is_training=is_training)

net = hk.transform_with_state(net)


def loss(params, net_state, x, y, is_training=True):
    out, net_state = net.apply(params, net_state, None, x, is_training=is_training)
    logits = out - logsumexp(out, axis=1).reshape(-1, 1)
    return -jnp.mean(y * logits), net_state


def accuracy(params, net_state, images, labels):
    s = 0
    for i in range(0, 1875):
        x, y = images[32*i: 32*(i+1)], labels[32*i: 32*(i+1)]
        true_class = jnp.argmax(y, axis=1)
        predicted_class = jnp.argmax(net.apply(params, net_state, None, x, False)[0], axis=1)
        s += jnp.sum(predicted_class == true_class)
    return s/60000
