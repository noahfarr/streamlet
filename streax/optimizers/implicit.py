from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from streax.utils import broadcast
from streax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class ImplicitConfig:
    lr: float
    eps: float = 1e-8
    clip_delta: bool = struct.field(pytree_node=False, default=True)
    adaptive_clip: bool = struct.field(pytree_node=False, default=True)
    normalize_delta: bool = struct.field(pytree_node=False, default=False)
    beta_clip: float = 0.9998
    beta_norm: float = 0.9998
    clip_multiplier: float = 20.0


@struct.dataclass(frozen=True)
class ImplicitState:
    squared_delta: Array
    absolute_delta: Array
    clip_step: Array
    norm_step: Array


@dataclass
class Implicit:

    cfg: ImplicitConfig
    name: str = "implicit"

    def init(self, parameters: PyTree, num_envs: int) -> ImplicitState:
        del parameters
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        return ImplicitState(
            squared_delta=jnp.ones((num_envs,), dtype=jnp.float32),
            absolute_delta=zeros,
            clip_step=jnp.int32(0),
            norm_step=jnp.int32(0),
        )

    def _safe_delta(
        self, state: ImplicitState, td_error: Array
    ) -> tuple[Array, ImplicitState]:
        cfg = self.cfg
        if not cfg.clip_delta:
            return td_error, state

        if cfg.adaptive_clip:
            clip_step = state.clip_step + 1
            squared_delta = (
                cfg.beta_clip * state.squared_delta
                + (1.0 - cfg.beta_clip) * td_error * td_error
            )
            ceiling = cfg.clip_multiplier * jnp.sqrt(
                squared_delta / (1.0 - cfg.beta_clip**clip_step)
            )
            clipped = jnp.sign(td_error) * jnp.minimum(jnp.abs(td_error), ceiling)
        else:
            clip_step = state.clip_step
            squared_delta = state.squared_delta
            clipped = jnp.clip(td_error, -1.0, 1.0)

        if cfg.normalize_delta:
            norm_step = state.norm_step + 1
            absolute_delta = (
                cfg.beta_norm * state.absolute_delta
                + (1.0 - cfg.beta_norm) * jnp.abs(clipped)
            )
            scale = absolute_delta / (1.0 - cfg.beta_norm**norm_step)
            safe_delta = clipped / jnp.maximum(scale, 1e-12)
        else:
            norm_step = state.norm_step
            absolute_delta = state.absolute_delta
            safe_delta = clipped

        state = ImplicitState(
            squared_delta=squared_delta,
            absolute_delta=absolute_delta,
            clip_step=clip_step,
            norm_step=norm_step,
        )
        return safe_delta, state

    def update(
        self,
        state: ImplicitState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array,
    ) -> tuple[PyTree, ImplicitState]:
        cfg = self.cfg

        safe_delta, state = self._safe_delta(state, td_error)

        squared_gradient_norm = sum(
            jnp.sum(jnp.square(g), axis=tuple(range(1, g.ndim)))
            for g in jax.tree.leaves(gradient)
        )
        effective_curvature = jnp.where(
            curvature > 0.0, curvature, squared_gradient_norm
        )
        denominator = jnp.maximum(1.0 + cfg.lr * effective_curvature, cfg.eps)
        step_size = jnp.minimum(cfg.lr / denominator, cfg.lr)

        def compute_update(trace_leaf):
            return (
                broadcast(step_size, trace_leaf)
                * broadcast(safe_delta, trace_leaf)
                * trace_leaf
            ).mean(axis=0)

        updates = jax.tree.map(compute_update, trace)

        lox.log(
            {
                f"{self.name}/step_size": step_size.mean(),
                f"{self.name}/curvature": curvature.mean(),
                f"{self.name}/denominator": denominator.mean(),
                f"{self.name}/safe_delta": safe_delta.mean(),
                f"{self.name}/update_norm": optax.global_norm(updates),
            }
        )

        return updates, state
