from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct

from streamlet.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class AutostepConfig:
    alpha_init: float = 0.1
    mu: float = 1e-2
    tau: float = 1e4
    eps: float = 1e-8
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class AutostepState:
    alpha: PyTree
    h: PyTree
    v: PyTree


@dataclass
class Autostep:

    cfg: AutostepConfig
    name: str = "autostep"

    def init(self, parameters: PyTree) -> AutostepState:
        alpha = jax.tree.map(
            lambda p: jnp.full(p.shape, self.cfg.alpha_init, dtype=self.cfg.dtype),
            parameters,
        )
        zeros = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=self.cfg.dtype), parameters
        )
        return AutostepState(alpha=alpha, h=zeros, v=zeros)

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        del state, gradient, trace, gamma, not_done
        return bootstrap_fn(params), None

    def update(
        self,
        state: AutostepState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array | None = None,
    ) -> tuple[PyTree, AutostepState]:
        del curvature
        cfg = self.cfg

        meta_gradient = jax.tree.map(
            lambda g, h: td_error * g * h, gradient, state.h
        )
        interaction = jax.tree.map(lambda a, g, z: a * g * z, state.alpha, gradient, trace)

        v = jax.tree.map(
            lambda v_leaf, m, i: jnp.maximum(
                jnp.abs(m), v_leaf + (jnp.abs(i) / cfg.tau) * (jnp.abs(m) - v_leaf)
            ).astype(cfg.dtype),
            state.v,
            meta_gradient,
            interaction,
        )

        alpha = jax.tree.map(
            lambda a, m, v_leaf: (
                a * jnp.exp(cfg.mu * jnp.where(v_leaf > 0.0, m / (v_leaf + cfg.eps), 0.0))
            ).astype(cfg.dtype),
            state.alpha,
            meta_gradient,
            v,
        )

        effective = sum(
            jnp.sum(a * jnp.abs(g * z))
            for a, g, z in zip(
                jax.tree.leaves(alpha), jax.tree.leaves(gradient), jax.tree.leaves(trace)
            )
        )
        M = jnp.maximum(effective, 1.0)
        alpha = jax.tree.map(lambda a: a / M, alpha)

        updates = jax.tree.map(
            lambda a, z: (a * td_error * z).astype(cfg.dtype), alpha, trace
        )

        h = jax.tree.map(
            lambda h_leaf, a, g, z, u: (
                h_leaf * jnp.clip(1.0 - a * g * z, 0.0, 1.0) + u
            ).astype(cfg.dtype),
            state.h,
            alpha,
            gradient,
            trace,
            updates,
        )

        alpha_leaves = jax.tree.leaves(alpha)
        alpha_mean = sum(jnp.sum(a) for a in alpha_leaves) / sum(
            a.size for a in alpha_leaves
        )

        lox.log(
            {
                f"{self.name}/step_size": alpha_mean,
                f"{self.name}/max_step_size": jnp.max(
                    jnp.stack([jnp.max(a) for a in alpha_leaves])
                ),
                f"{self.name}/effective_step_size": effective,
            }
        )

        return updates, AutostepState(alpha=alpha, h=h, v=v)
