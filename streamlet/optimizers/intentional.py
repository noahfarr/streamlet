from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct
from streamlet.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class IntentionalConfig:
    gamma: float
    trace_lambda: float
    eta: float = 0.5
    beta2: float = 0.999
    beta_clip: float = 0.9998
    beta_norm: float = 0.9998
    clip_multiplier: float = 20.0
    eps: float = 1e-8
    normalize_delta: bool = struct.field(pytree_node=False, default=False)
    use_adaptive_clip: bool = struct.field(pytree_node=False, default=True)
    use_rmsprop: bool = struct.field(pytree_node=False, default=True)
    use_sigma: bool = struct.field(pytree_node=False, default=True)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class IntentionalState:
    second_moment: PyTree
    sigma: Array
    squared_delta_ema: Array
    absolute_delta_ema: Array
    clip_step: Array
    norm_step: Array
    step: Array


@dataclass
class Intentional:
    cfg: IntentionalConfig
    name: str = "intentional"

    def init(self, parameters: PyTree) -> IntentionalState:
        second_moment = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=self.cfg.dtype),
            parameters,
        )
        return IntentionalState(
            second_moment=second_moment,
            sigma=jnp.float32(0.0),
            squared_delta_ema=jnp.float32(1.0),
            absolute_delta_ema=jnp.float32(0.0),
            clip_step=jnp.int32(0),
            norm_step=jnp.int32(0),
            step=jnp.int32(0),
        )

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        return bootstrap_fn(params), None

    def update(
        self,
        state: IntentionalState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array | None = None,
    ) -> tuple[PyTree, IntentionalState]:
        cfg = self.cfg
        next_step = state.step + 1

        new_second_moment = jax.tree.map(
            lambda v, g: (cfg.beta2 * v + (1.0 - cfg.beta2) * jnp.square(g)).astype(
                cfg.dtype
            ),
            state.second_moment,
            gradient,
        )

        if cfg.use_rmsprop:
            preconditioner = jax.tree.map(
                lambda v: jnp.sqrt(v / (1.0 - cfg.beta2**next_step)) + cfg.eps,
                new_second_moment,
            )
        else:
            preconditioner = jax.tree.map(jnp.ones_like, new_second_moment)

        squared_gradient_norm = sum(
            jnp.sum(jnp.square(g) / m)
            for g, m in zip(jax.tree.leaves(gradient), jax.tree.leaves(preconditioner))
        )
        squared_trace_norm = sum(
            jnp.sum(jnp.square(t) / m)
            for t, m in zip(jax.tree.leaves(trace), jax.tree.leaves(preconditioner))
        )

        gamma_lambda = cfg.gamma * cfg.trace_lambda
        if cfg.use_sigma:
            new_sigma = state.sigma + (1.0 - gamma_lambda) * (
                squared_gradient_norm - state.sigma
            )
            sigma_unbiased = new_sigma / (1.0 - gamma_lambda**next_step)
            denominator = jnp.sqrt(sigma_unbiased * squared_trace_norm)
        else:
            new_sigma = state.sigma
            denominator = squared_trace_norm

        step_size = cfg.eta / jnp.maximum(denominator, cfg.eps)

        if cfg.use_adaptive_clip:
            next_clip_step = state.clip_step + 1
            new_squared_delta_ema = (
                cfg.beta_clip * state.squared_delta_ema
                + (1.0 - cfg.beta_clip) * td_error * td_error
            )
            clip_ceiling = cfg.clip_multiplier * jnp.sqrt(
                new_squared_delta_ema / (1.0 - cfg.beta_clip**next_clip_step)
            )
            clipped_delta = jnp.sign(td_error) * jnp.minimum(
                jnp.abs(td_error), clip_ceiling
            )
        else:
            next_clip_step = state.clip_step
            new_squared_delta_ema = state.squared_delta_ema
            clipped_delta = jnp.clip(td_error, -1.0, 1.0)

        if cfg.normalize_delta:
            next_norm_step = state.norm_step + 1
            new_absolute_delta_ema = cfg.beta_norm * state.absolute_delta_ema + (
                1.0 - cfg.beta_norm
            ) * jnp.abs(clipped_delta)
            absolute_delta_unbiased = new_absolute_delta_ema / (
                1.0 - cfg.beta_norm**next_norm_step
            )
            safe_delta = clipped_delta / jnp.maximum(absolute_delta_unbiased, 1e-12)
        else:
            next_norm_step = state.norm_step
            new_absolute_delta_ema = state.absolute_delta_ema
            safe_delta = clipped_delta

        scale = safe_delta * step_size
        updates = jax.tree.map(
            lambda t, m: (scale * t / m).astype(cfg.dtype),
            trace,
            preconditioner,
        )

        new_state = IntentionalState(
            second_moment=new_second_moment,
            sigma=new_sigma,
            squared_delta_ema=new_squared_delta_ema,
            absolute_delta_ema=new_absolute_delta_ema,
            clip_step=next_clip_step,
            norm_step=next_norm_step,
            step=next_step,
        )
        lox.log(
            {
                f"{self.name}/step_size": step_size.mean(),
                f"{self.name}/denominator": denominator.mean(),
                f"{self.name}/sigma": new_sigma.mean(),
                f"{self.name}/trace_norm": jnp.sqrt(squared_trace_norm).mean(),
            }
        )
        return updates, new_state
