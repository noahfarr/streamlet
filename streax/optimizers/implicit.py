from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from streax.utils import broadcast
from streax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class ImplicitConfig:
    """Implicit TD(lambda): the Intentional backbone with a closed-loop term.

    The update is

        w <- w + eta * delta_tilde * (rho z) / ( sqrt(sigma_bar <z, rho z>) + eta * c_t )

    with c_t = <g - gamma g', rho z> the interaction (curvature) supplied by the
    algorithm, and the denominator floored at kappa * sqrt(sigma_bar <z, rho z>)
    so a negative c_t cannot shrink or flip it. Setting c_t -> 0 recovers
    Intentional exactly; rho = (sqrt(nu) + eps)^-1 is the RMSProp preconditioner
    and sigma_bar the discounted accumulated gradient energy E[<g, rho g>].
    """

    gamma: float
    trace_lambda: float
    eta: float = 0.5
    kappa: float = 1.0
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
class ImplicitState:
    second_moment: PyTree
    sigma: Array
    squared_delta_ema: Array
    absolute_delta_ema: Array
    clip_step: Array
    norm_step: Array
    step: Array


@dataclass
class Implicit:
    cfg: ImplicitConfig
    name: str = "implicit"

    def init(self, parameters: PyTree, num_envs: int) -> ImplicitState:
        second_moment = jax.tree.map(
            lambda p: jnp.ones((num_envs, *p.shape), dtype=self.cfg.dtype),
            parameters,
        )
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        return ImplicitState(
            second_moment=second_moment,
            sigma=zeros,
            squared_delta_ema=jnp.ones((num_envs,), dtype=jnp.float32),
            absolute_delta_ema=zeros,
            clip_step=jnp.int32(0),
            norm_step=jnp.int32(0),
            step=jnp.int32(0),
        )

    def precondition(self, state: ImplicitState, trace: PyTree) -> PyTree:
        if not self.cfg.use_rmsprop:
            return trace
        return jax.tree.map(
            lambda v, t: t / (jnp.sqrt(v) + self.cfg.eps), state.second_moment, trace
        )

    def update(
        self,
        state: ImplicitState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array,
        squared_grad_norm: Array | None = None,
        *,
        td_error_grad: PyTree | None = None,
        h_value: Array | None = None,
        bias_trace: Array | None = None,
    ) -> tuple[PyTree, ImplicitState]:
        del squared_grad_norm
        qrc = td_error_grad is not None
        cfg = self.cfg
        next_step = state.step + 1

        if cfg.use_rmsprop:
            preconditioner = jax.tree.map(
                lambda v: jnp.sqrt(v) + cfg.eps, state.second_moment
            )
        else:
            preconditioner = jax.tree.map(jnp.ones_like, state.second_moment)

        new_second_moment = jax.tree.map(
            lambda v, g: (cfg.beta2 * v + (1.0 - cfg.beta2) * jnp.square(g)).astype(
                cfg.dtype
            ),
            state.second_moment,
            gradient,
        )

        squared_gradient_norm = sum(
            jnp.sum(jnp.square(g) / m, axis=tuple(range(1, g.ndim)))
            for g, m in zip(jax.tree.leaves(gradient), jax.tree.leaves(preconditioner))
        )
        squared_trace_norm = sum(
            jnp.sum(jnp.square(t) / m, axis=tuple(range(1, t.ndim)))
            for t, m in zip(jax.tree.leaves(trace), jax.tree.leaves(preconditioner))
        )

        gamma_lambda = cfg.gamma * cfg.trace_lambda
        if cfg.use_sigma:
            new_sigma = state.sigma + (1.0 - gamma_lambda) * (
                squared_gradient_norm - state.sigma
            )
            sigma_unbiased = new_sigma / (1.0 - gamma_lambda**next_step)
            baseline = jnp.sqrt(sigma_unbiased * squared_trace_norm)
        else:
            new_sigma = state.sigma
            baseline = squared_trace_norm

        denominator = baseline + cfg.eta * curvature
        denominator = jnp.maximum(denominator, cfg.kappa * baseline)
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

        if qrc:
            bg = sum(
                jnp.sum(b * g / m, axis=tuple(range(1, b.ndim)))
                for b, g, m in zip(
                    jax.tree.leaves(td_error_grad),
                    jax.tree.leaves(gradient),
                    jax.tree.leaves(preconditioner),
                )
            )
            bb = sum(
                jnp.sum(b * b / m, axis=tuple(range(1, b.ndim)))
                for b, m in zip(
                    jax.tree.leaves(td_error_grad),
                    jax.tree.leaves(preconditioner),
                )
            )
            base_step = cfg.eta / jnp.maximum(baseline, cfg.eps)
            proximal_delta = safe_delta - base_step * (h_value * bg + bias_trace * bb)

            scale_z = proximal_delta * step_size
            scale_g = h_value * base_step
            scale_b = bias_trace * base_step
            updates = jax.tree.map(
                lambda z, g, b, m: (
                    (
                        broadcast(scale_z, z) * z
                        - broadcast(scale_g, g) * g
                        - broadcast(scale_b, b) * b
                    )
                    / m
                )
                .mean(axis=0)
                .astype(cfg.dtype),
                trace,
                gradient,
                td_error_grad,
                preconditioner,
            )
        else:
            scale = safe_delta * step_size
            updates = jax.tree.map(
                lambda t, m: (broadcast(scale, t) * t / m)
                .mean(axis=0)
                .astype(cfg.dtype),
                trace,
                preconditioner,
            )

        new_state = ImplicitState(
            second_moment=new_second_moment,
            sigma=new_sigma,
            squared_delta_ema=new_squared_delta_ema,
            absolute_delta_ema=new_absolute_delta_ema,
            clip_step=next_clip_step,
            norm_step=next_norm_step,
            step=next_step,
        )
        log_dict = {
            f"{self.name}/step_size": step_size.mean(),
            f"{self.name}/curvature": curvature.mean(),
            f"{self.name}/denominator": denominator.mean(),
            f"{self.name}/baseline": baseline.mean(),
            f"{self.name}/sigma": new_sigma.mean(),
            f"{self.name}/safe_delta": safe_delta.mean(),
        }
        if qrc:
            log_dict[f"{self.name}/proximal_delta"] = proximal_delta.mean()
            log_dict[f"{self.name}/bg"] = bg.mean()
            log_dict[f"{self.name}/bb"] = bb.mean()
        lox.log(log_dict)
        return updates, new_state
