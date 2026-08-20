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
    diagonal: bool = struct.field(pytree_node=False, default=False)
    prior_scale: float = 1.0
    sigma_mode: str = struct.field(pytree_node=False, default="residual")
    precondition: bool = struct.field(pytree_node=False, default=False)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class KalmanState:
    m: Array
    delta_sq_ema: Array
    dg_sq_ema: Array
    dim: Array
    step: Array
    delta_cross_ema: Array
    previous_delta: Array


@dataclass
class Kalman:
    cfg: KalmanConfig
    name: str = "kalman"

    def init(self, parameters: PyTree) -> KalmanState:
        dim = sum(leaf.size for leaf in jax.tree.leaves(parameters))
        if self.cfg.diagonal:
            m = jax.tree.map(
                lambda p: jnp.full_like(
                    p,
                    self.cfg.prior_scale
                    * (jnp.mean(jnp.square(p)) + self.cfg.eps),
                ),
                parameters,
            )
        else:
            m = jnp.float32(self.cfg.m_init)
        return KalmanState(
            m=m,
            delta_sq_ema=jnp.float32(0.0),
            dg_sq_ema=jnp.float32(0.0),
            dim=jnp.float32(dim),
            step=jnp.int32(0),
            delta_cross_ema=jnp.float32(0.0),
            previous_delta=jnp.float32(0.0),
        )

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        next_value, pullback = jax.vjp(bootstrap_fn, params)
        (next_gradient,) = pullback(jnp.ones_like(next_value))
        delta_g = jax.tree.map(
            lambda g, gn: g - gamma * not_done * gn, gradient, next_gradient
        )
        if self.cfg.diagonal:
            return next_value, delta_g
        interaction = sum(
            jnp.sum(d * z)
            for d, z in zip(jax.tree.leaves(delta_g), jax.tree.leaves(trace))
        )
        dg_sq = sum(jnp.sum(jnp.square(d)) for d in jax.tree.leaves(delta_g))
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
        trace_sq = sum(jnp.sum(jnp.square(z)) for z in jax.tree.leaves(trace))

        next_step = state.step + 1
        correction = 1.0 - cfg.beta**next_step
        delta_sq_ema = cfg.beta * state.delta_sq_ema + (1.0 - cfg.beta) * jnp.square(
            td_error
        )
        delta_sq_hat = delta_sq_ema / correction
        delta_cross_ema = cfg.beta * state.delta_cross_ema + (1.0 - cfg.beta) * (
            td_error * state.previous_delta
        )
        delta_cross_hat = delta_cross_ema / correction

        if cfg.diagonal:
            delta_g = curvature
            v = sum(
                jnp.sum(m * jnp.square(d))
                for m, d in zip(jax.tree.leaves(state.m), jax.tree.leaves(delta_g))
            )
            weighted_interaction = sum(
                jnp.sum(m * z * d)
                for m, z, d in zip(
                    jax.tree.leaves(state.m),
                    jax.tree.leaves(trace),
                    jax.tree.leaves(delta_g),
                )
            )
            dg_sq_ema = cfg.beta * state.dg_sq_ema + (1.0 - cfg.beta) * v
            v_hat = dg_sq_ema / correction
            if cfg.sigma_mode == "autocovariance":
                sigma_sq = jnp.maximum(
                    delta_sq_hat - jnp.abs(delta_cross_hat),
                    cfg.sigma_floor * delta_sq_hat,
                )
            else:
                sigma_sq = jnp.maximum(
                    delta_sq_hat - v_hat, cfg.sigma_floor * delta_sq_hat
                )
            if cfg.precondition:
                weighted_interaction = sum(
                    jnp.sum(jnp.square(m) * z * d)
                    for m, z, d in zip(
                        jax.tree.leaves(state.m),
                        jax.tree.leaves(trace),
                        jax.tree.leaves(delta_g),
                    )
                )
                direction_sq = sum(
                    jnp.sum(jnp.square(m * z))
                    for m, z in zip(
                        jax.tree.leaves(state.m), jax.tree.leaves(trace)
                    )
                )
            else:
                direction_sq = trace_sq
            gain = jnp.maximum(weighted_interaction, 0.0)
            denominator = (v + sigma_sq) * direction_sq + cfg.eps
            alpha = jnp.minimum(gain / denominator, cfg.alpha_max)
            m = jax.tree.map(
                lambda mi, d: jnp.maximum(
                    mi
                    - jnp.square(mi * d) / (v + sigma_sq + cfg.eps)
                    + cfg.process_noise,
                    cfg.eps,
                ),
                state.m,
                delta_g,
            )
            interaction = weighted_interaction
        else:
            interaction = curvature[0]
            dg_sq = curvature[1]
            dg_sq_ema = cfg.beta * state.dg_sq_ema + (1.0 - cfg.beta) * dg_sq
            dg_sq_hat = dg_sq_ema / correction
            if cfg.sigma_mode == "autocovariance":
                sigma_sq = jnp.maximum(
                    delta_sq_hat - jnp.abs(delta_cross_hat),
                    cfg.sigma_floor * delta_sq_hat,
                )
            else:
                sigma_sq = jnp.maximum(
                    delta_sq_hat - state.m * dg_sq_hat, cfg.sigma_floor * delta_sq_hat
                )
            gain = state.m * jnp.maximum(interaction, 0.0)
            denominator = (state.m * dg_sq + sigma_sq) * trace_sq + cfg.eps
            alpha = jnp.minimum(gain / denominator, cfg.alpha_max)
            reduction = (
                jnp.square(alpha) * jnp.square(td_error) * trace_sq / state.dim
            )
            m = jnp.maximum(state.m - reduction + cfg.process_noise, cfg.eps)

        scale = alpha * td_error
        if cfg.diagonal and cfg.precondition:
            updates = jax.tree.map(
                lambda m, z: (scale * m * z).astype(cfg.dtype), state.m, trace
            )
        else:
            updates = jax.tree.map(lambda z: (scale * z).astype(cfg.dtype), trace)

        new_state = KalmanState(
            m=m,
            delta_sq_ema=delta_sq_ema,
            dg_sq_ema=dg_sq_ema,
            dim=state.dim,
            step=next_step,
            delta_cross_ema=delta_cross_ema,
            previous_delta=td_error,
        )
        lox.log(
            {
                f"{self.name}/step_size": alpha,
                f"{self.name}/m": (
                    sum(jnp.sum(leaf) for leaf in jax.tree.leaves(m)) / state.dim
                    if cfg.diagonal
                    else m
                ),
                f"{self.name}/sigma_sq": sigma_sq,
                f"{self.name}/delta_sq": delta_sq_hat,
                f"{self.name}/delta_cross": delta_cross_hat,
                f"{self.name}/interaction": interaction,
                f"{self.name}/trace_sq": trace_sq,
            }
        )
        return updates, new_state
