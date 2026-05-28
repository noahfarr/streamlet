from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from stremax.utils import broadcast
from stremax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class ImplicitConfig:
    lr: float
    eps: float = 1e-8


@struct.dataclass(frozen=True)
class ImplicitState:
    pass


@dataclass
class Implicit:

    cfg: ImplicitConfig
    name: str = "implicit"

    def init(self, parameters: PyTree, num_envs: int) -> ImplicitState:
        del parameters, num_envs
        return ImplicitState()

    def update(
        self,
        state: ImplicitState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array,
    ) -> tuple[PyTree, ImplicitState]:
        cfg = self.cfg

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
                * broadcast(td_error, trace_leaf)
                * trace_leaf
            ).mean(axis=0)

        updates = jax.tree.map(compute_update, trace)

        lox.log(
            {
                f"{self.name}/step_size": step_size.mean(),
                f"{self.name}/curvature": curvature.mean(),
                f"{self.name}/denominator": denominator.mean(),
                f"{self.name}/update_norm": optax.global_norm(updates),
            }
        )

        return updates, state
