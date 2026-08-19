from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct

from streamlet.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class KalmanConfig:
    gamma: float
    m_init: float = 1.0
    process_noise: float = 1e-6
    beta: float = 0.999
    sigma_floor: float = 0.1
    alpha_max: float = 1.0
    eps: float = 1e-8
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class KalmanState:
    m: Array
    delta_sq_ema: Array
    dg_sq_ema: Array
    dim: Array
    step: Array


@dataclass
class Kalman:
    cfg: KalmanConfig
    name: str = "kalman"

    def init(self, parameters: PyTree) -> KalmanState:
        dim = sum(leaf.size for leaf in jax.tree.leaves(parameters))
        return KalmanState(
            m=jnp.float32(self.cfg.m_init),
            delta_sq_ema=jnp.float32(0.0),
            dg_sq_ema=jnp.float32(0.0),
            dim=jnp.float32(dim),
            step=jnp.int32(0),
        )

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        gradient_trace = sum(
            jnp.sum(g * z)
            for g, z in zip(jax.tree.leaves(gradient), jax.tree.leaves(trace))
        )
        next_value, pullback = jax.vjp(bootstrap_fn, params)
        (next_gradient,) = pullback(jnp.ones_like(next_value))
        next_grad_trace = sum(
            jnp.sum(g * z)
            for g, z in zip(jax.tree.leaves(next_gradient), jax.tree.leaves(trace))
        )
        interaction = gradient_trace - gamma * not_done * next_grad_trace
        dg_sq = sum(
            jnp.sum(jnp.square(g - gamma * not_done * gn))
            for g, gn in zip(
                jax.tree.leaves(gradient), jax.tree.leaves(next_gradient)
            )
        )
        return next_value, jnp.stack([interaction, dg_sq])

    def update(
        self,
        state: KalmanState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array,
    ) -> tuple[PyTree, KalmanState]:
        cfg = self.cfg
        interaction = curvature[0]
        dg_sq = curvature[1]

        trace_sq = sum(jnp.sum(jnp.square(z)) for z in jax.tree.leaves(trace))

        next_step = state.step + 1
        correction = 1.0 - cfg.beta**next_step
        delta_sq_ema = cfg.beta * state.delta_sq_ema + (1.0 - cfg.beta) * jnp.square(
            td_error
        )
        dg_sq_ema = cfg.beta * state.dg_sq_ema + (1.0 - cfg.beta) * dg_sq
        delta_sq_hat = delta_sq_ema / correction
        dg_sq_hat = dg_sq_ema / correction

        sigma_sq = jnp.maximum(
            delta_sq_hat - state.m * dg_sq_hat, cfg.sigma_floor * delta_sq_hat
        )

        gain = state.m * jnp.maximum(interaction, 0.0)
        denominator = (state.m * dg_sq + sigma_sq) * trace_sq + cfg.eps
        alpha = jnp.minimum(gain / denominator, cfg.alpha_max)

        scale = alpha * td_error
        updates = jax.tree.map(lambda z: (scale * z).astype(cfg.dtype), trace)

        reduction = jnp.square(alpha) * jnp.square(td_error) * trace_sq / state.dim
        m = jnp.maximum(state.m - reduction + cfg.process_noise, cfg.eps)

        new_state = KalmanState(
            m=m,
            delta_sq_ema=delta_sq_ema,
            dg_sq_ema=dg_sq_ema,
            dim=state.dim,
            step=next_step,
        )
        lox.log(
            {
                f"{self.name}/step_size": alpha,
                f"{self.name}/m": m,
                f"{self.name}/sigma_sq": sigma_sq,
                f"{self.name}/interaction": interaction,
                f"{self.name}/dg_sq": dg_sq,
                f"{self.name}/trace_sq": trace_sq,
            }
        )
        return updates, new_state
