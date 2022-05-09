from itertools import accumulate
from jax import jit, grad, tree_map
from jax.tree_util import tree_reduce
from model import loss
import jax.numpy as jnp
import attrs


@attrs.define
class Optimizer:
    def create_state(self, params):
        pass

    def on_epoch_state_update(params, state, batch):
        return state

    def on_step_state_update(params, state, batch):
        return state

    def update(params, state, batch):
        return params, state


@attrs.define
class SGD(Optimizer):
    step_size: float = 0.01

    def create_state(self, params):
        return {"step_size": self.step_size}

    @jit
    def update(params, state, batch):
        x, y = batch
        step_size = state["step_size"]

        grads = grad(loss)(params, x, y)
        return tree_map(lambda p, g: p - step_size * g, params, grads), state


@attrs.define
class AdaGrad(Optimizer):

    eta: float = 0.01
    epsilon: float = 1e-7

    def create_state(self, params):
        G = tree_map(lambda p: jnp.zeros_like(p) + self.epsilon, params)
        return {"eta": self.eta, "G": G}

    @jit
    def on_step_state_update(params, state, batch):
        grads = grad(loss)(params, x, y)
        state["G"] = tree_map(lambda prev, g: prev + g**2, state["G"], grads)
        return state

    @jit
    def update(params, state, batch):
        x, y = batch
        eta = state["eta"]

        grads = grad(loss)(params, x, y)

        updated_grads = tree_map(
            lambda precond, g: g / jnp.sqrt(precond), state["G"], grads
        )

        return tree_map(lambda p, g: p - eta * g, params, updated_grads), state


@attrs.define
class SpiderBoost(Optimizer):
    step_size: float = 0.01

    def create_state(self, params):
        v = tree_map(lambda p: jnp.zeros_like(p), params)
        prev_params = tree_map(lambda p: jnp.zeros_like(p), params)
        return {"step_size": self.step_size, "V": v, "prev_params": prev_params}
    
    @jit
    def on_epoch_state_update(params, state, batch):
        x, y = batch
        grads = grad(loss)(params, x, y)
        state["V"] = grads
        return state

    @jit
    def on_step_state_update(params, state, batch):
        x, y = batch
        prev_params = state["prev_params"]

        grads_prev = grad(loss)(prev_params, x, y)
        grads = grad(loss)(params, x, y)

        state["V"] = tree_map(
            lambda curr_grad, prev_grad, v: curr_grad - prev_grad + v,
            grads,
            grads_prev,
            state["V"],
        )
        return state


    @jit
    def update(params, state, batch):
        x, y = batch
        step_size = state["step_size"]
        V = state['V']

        state['prev_params'] = params
        
        return tree_map(lambda p, g: p - step_size * g, params, V), state


@attrs.define
class AdaSpider(Optimizer):
    eta: float = 1.0
    n: int = 0

    def create_state(self, params):
        v = tree_map(lambda p: jnp.zeros_like(p), params)
        prev_params = tree_map(lambda p: jnp.zeros_like(p), params)
        return {
            "V": v,
            "prev_params": prev_params,
            "acc_v": 0.0,
            "n": self.n,
            "eta": self.eta,
            "step_size": 0.0
        }

    @jit
    def on_step_state_update(params, state, batch):
        x, y = batch
        V = state["V"]
        prev_params = state["prev_params"]
        accumulated_norms = state["acc_v"]

        grads_prev = grad(loss)(prev_params, x, y)
        grads = grad(loss)(params, x, y)

        V = tree_map(
            lambda curr_grad, prev_grad, v: curr_grad - prev_grad + v,
            grads,
            grads_prev,
            V,
        )

        norms = tree_map(lambda v: jnp.sum(v * v), V)
        total_norm = tree_reduce(lambda a, b: a + b, norms, 0.0)

        accumulated_norms += total_norm

        state["V"] = V
        state["acc_v"] = accumulated_norms

        return state

    @jit
    def update(params, state, batch):
        V = state["V"]
        accumulated_norms = state["acc_v"]
        n = state["n"]
        eta = state["eta"]

        state["prev_params"] = tree_map(lambda x: x.copy(), params)

        step_size = 1 / (n ** (1 / 4) * jnp.sqrt(jnp.sqrt(n) + accumulated_norms))
        state["step_size"] = step_size
        return tree_map(lambda p, g: p - eta * step_size * g, params, V), state

    @jit
    def on_epoch_state_update(params, state, batch):
        x, y = batch
        grads = grad(loss)(params, x, y)
        accumulated_norms = state["acc_v"]

        state["V"] = grads
        
        norms = tree_map(lambda v: jnp.sum(v * v), state["V"])
        total_norm = tree_reduce(lambda a, b: a + b, norms, 0.0)

        accumulated_norms += total_norm
        state["acc_v"] = accumulated_norms
        state["prev_params"] = tree_map(lambda x: x.copy(), params)
        return state
