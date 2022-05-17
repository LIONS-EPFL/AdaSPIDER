from itertools import accumulate
from xml.etree.ElementTree import XMLParser
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
        return params, state

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
        x, y = batch
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
    L: float = 1.0

    def create_state(self, params):
        v = tree_map(lambda p: jnp.zeros_like(p), params)
        prev_params = tree_map(lambda p: jnp.zeros_like(p), params)
        return {"step_size": 1.0/(2*self.L), "V": v, "prev_params": prev_params}
    
    @jit
    def on_epoch_state_update(params, state, batch):
        x, y = batch
        grads = grad(loss)(params, x, y)
        state["V"] = grads
        state["prev_params"] = params
        return params, state

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
        state["prev_params"] = tree_map(lambda z: z.copy(), params)
        return params, state



@attrs.define
class AdaSpiderBoost(Optimizer):
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
        eta = state["eta"]

        state["prev_params"] = tree_map(lambda x: x.copy(), params)

        step_size = 1 / (jnp.sqrt(1 + accumulated_norms))
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
        state["prev_params"] = tree_map(lambda z: z.copy(), params)
        return params, state


@attrs.define
class Spider(Optimizer):
    epsilon: float = 0.01
    L: float = 1
    n_zero: int = 0

    def create_state(self, params):
        v = tree_map(lambda p: jnp.zeros_like(p), params)
        prev_params = tree_map(lambda p: jnp.zeros_like(p), params)
        return {
            "V": v,
            "prev_params": prev_params,
            "n_zero": self.n_zero,
            "epsilon": self.epsilon,
            "L": self.L,
            "step_size": 0.0
        }
    
    @jit
    def on_step_state_update(params, state, batch):
        x, y = batch
        V = state["V"]
        prev_params = state["prev_params"]

        grads_prev = grad(loss)(prev_params, x, y)
        grads = grad(loss)(params, x, y)

        V = tree_map(
            lambda curr_grad, prev_grad, v: curr_grad - prev_grad + v,
            grads,
            grads_prev,
            V,
        )
        state["V"] = V
        return state

    @jit
    def update(params, state, batch):
        V = state["V"]
        L = state["L"]
        n_zero = state["n_zero"]
        epsilon = state["epsilon"]

        state["prev_params"] = tree_map(lambda x: x.copy(), params)

        norms = tree_map(lambda v: jnp.sum(v * v), V)
        norm_V = jnp.sqrt(tree_reduce(lambda a, b: a + b, norms, 0.0))

        step_size = epsilon / (L * n_zero * norm_V)
        state["step_size"] = step_size
        return tree_map(lambda p, g: p - step_size * g, params, V), state

    @jit
    def on_epoch_state_update(params, state, batch):
        x, y = batch
        grads = grad(loss)(params, x, y)

        state["V"] = grads
  
        state["prev_params"] = tree_map(lambda z: z.copy(), params)
        return params, state


@attrs.define
class KatyushaXw(Optimizer):
    step_size: float = 0.01


    def create_state(self, params):
        nabla = tree_map(lambda p: jnp.zeros_like(p), params)
        x = tree_map(lambda p: p.copy(), params)
        y = tree_map(lambda p: p.copy(), params)
        return {
            "nabla": nabla,
            "x": x,
            "prev_y": y,
            "k": 0,
            "step_size": self.step_size
        }
    
    @jit
    def on_epoch_state_update(params, state, batch):
        xs = state["x"]
        k = state["k"]
        prev_ys = state["prev_y"]

        xs = tree_map(lambda x, y, prev_y: ((3*k + 1)*y + (k+1)*x - (2*k - 2)*prev_y)/(2*k + 4), xs, params, prev_ys)

        state["mu"] = grad(loss)(xs, batch[0], batch[1])

        state["x"] = xs 
        state["prev_y"] = tree_map(lambda p: p.copy(), params)
        state["k"] = state["k"] + 1

        return xs, state
    
    @jit
    def on_step_state_update(params, state, batch):
        patterns, labels = batch

        grads_prev = grad(loss)(state["x"], patterns, labels)
        grads = grad(loss)(params, patterns, labels)

        state["nabla"] = tree_map(
            lambda curr_grad, prev_grad, v: curr_grad - prev_grad + v,
            grads,
            grads_prev,
            state["mu"],
        )
        return state

    @jit
    def update(params, state, batch):
        step_size = state["step_size"]
        return tree_map(lambda p, g: p - step_size * g, params, state["nabla"]), state



@attrs.define
class AdaSpiderDiag(Optimizer):
    eta: float = 1.0
    n: int = 0

    def create_state(self, params):
        v = tree_map(lambda p: jnp.zeros_like(p), params)
        prev_params = tree_map(lambda p: jnp.zeros_like(p), params)
        norms = tree_map(lambda p: jnp.zeros_like(p), params)
        return {
            "V": v,
            "prev_params": prev_params,
            "norms": norms,
            "n": self.n,
            "eta": self.eta,
            "step_size": 0.0
        }

    @jit
    def on_step_state_update(params, state, batch):
        x, y = batch
        V = state["V"]
        prev_params = state["prev_params"]
        norms = state["norms"]

        grads_prev = grad(loss)(prev_params, x, y)
        grads = grad(loss)(params, x, y)

        V = tree_map(
            lambda curr_grad, prev_grad, v: curr_grad - prev_grad + v,
            grads,
            grads_prev,
            V,
        )

        current_norms = tree_map(lambda v: (v * v), V)
        norms = tree_map(lambda a, b: a + b, current_norms, norms)

        state["V"] = V
        state["norms"] = norms

        return state

    @jit
    def update(params, state, batch):
        V = state["V"]
        norms = state["norms"]
        n = state["n"]
        eta = state["eta"]

        state["prev_params"] = tree_map(lambda x: x.copy(), params)

        step_size = lambda nrm: 1 / (n ** (1 / 4) * jnp.sqrt(jnp.sqrt(n) + nrm))
        
        return tree_map(lambda p, g, nrm: p - eta * step_size(nrm) * g, params, V, norms), state

    @jit
    def on_epoch_state_update(params, state, batch):
        x, y = batch
        grads = grad(loss)(params, x, y)
        norms = state["norms"]

        state["V"] = grads
        
        current_norms = tree_map(lambda v: (v * v), state["V"])
        norms = tree_map(lambda a, b: a + b, norms, current_norms)

        state["norms"] = norms
        state["prev_params"] = tree_map(lambda z: z.copy(), params)
        return params, state


@attrs.define
class AdaSVRG(Optimizer):
    eta: float = 1.0


    def create_state(self, params):
        mu = tree_map(lambda p: jnp.zeros_like(p), params)
        grad_est = tree_map(lambda p: jnp.zeros_like(p), params)
        x = tree_map(lambda p: p.copy(), params)
        y = tree_map(lambda p: p.copy(), params)
        return {
            "mu": mu,
            "anchor": x,
            "acc_norm": 0.0,
            "step_size": 0.0,
            "eta": self.eta,
            "grad_est": grad_est
        }
    
    @jit
    def on_epoch_state_update(params, state, batch):
        state["anchor"] =  tree_map(lambda p: p.copy(), params)
        state["mu"] = grad(loss)(state["anchor"], batch[0], batch[1])

        norm = tree_map(lambda v: jnp.sum(v * v), state["mu"])
        state["acc_norm"] = tree_reduce(lambda a, b: a+b, norm, 0.0)
        
        return params, state
    
    @jit
    def on_step_state_update(params, state, batch):
        patterns, labels = batch

        grads_prev = grad(loss)(state["anchor"], patterns, labels)
        grads = grad(loss)(params, patterns, labels)

        state["grad_est"] = tree_map(
            lambda curr_grad, prev_grad, mu: curr_grad - prev_grad + mu,
            grads,
            grads_prev,
            state["mu"],
        )
        norm = tree_map(lambda mu: jnp.sum(mu * mu), state["grad_est"])
        state["acc_norm"] = tree_reduce(lambda a, b: a+b, norm, state["acc_norm"])
        return state

    @jit
    def update(params, state, batch):
        gamma = 1/jnp.sqrt(state["acc_norm"])
        eta = state["eta"]
        state["step_size"] = gamma
        return tree_map(lambda p, g: p - eta * gamma * g, params, state["grad_est"]), state





@attrs.define
class Adam(Optimizer):

    eta: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8

    def create_state(self, params):
        V = tree_map(lambda p: jnp.zeros_like(p), params)
        M = tree_map(lambda p: jnp.zeros_like(p), params)
        return {"eta": self.eta,
                "beta_1": self.beta_1,
                "beta_2": self.beta_2,
                "beta_1_pow_t": self.beta_1,
                "beta_2_pow_t": self.beta_2,
                "M": M,
                "V": V,
                "epsilon": self.epsilon}

    @jit
    def on_step_state_update(params, state, batch):
        x, y = batch
        beta_1, beta_2 = state["beta_1"], state["beta_2"]
        beta_1_pow_t, beta_2_pow_t = state["beta_1_pow_t"], state["beta_2_pow_t"]

        grads = grad(loss)(params, x, y)
        state["M"] = tree_map(lambda prev, g: beta_1 * prev + (1-beta_1)* g, state["M"], grads)
        state["V"] = tree_map(lambda prev, g: beta_2 * prev + (1-beta_2) * g**2, state["V"], grads)

        state["M"] = tree_map(lambda m: m/(1 - beta_1_pow_t), state["M"])
        state["V"] = tree_map(lambda v: v/(1 - beta_2_pow_t), state["V"])

        state["beta_1_pow_t"] = beta_1*beta_1_pow_t
        state["beta_2_pow_t"] = beta_2*beta_2_pow_t
        return state

    @jit
    def update(params, state, batch):
        eta = state["eta"]
        M = state["M"]
        V = state["V"]
        epsilon = state["epsilon"]

        return tree_map(lambda p, m, v: p - eta * m/(jnp.sqrt(v) + epsilon), params, M, V), state



