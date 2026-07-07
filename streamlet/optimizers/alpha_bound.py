from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct

from streamlet.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class AlphaBoundConfig:
    alpha_init: float = 1.0
    eps: float = 1e-8
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class AlphaBoundState:
    alpha: Array


@dataclass
class AlphaBound:
    """AlphaBound (Dabney & Barto, 2012), "Adaptive Step-Size for Online TD Learning".

    A single step size that only ever decreases. Each step it is clamped to the largest
    value that keeps the update from overshooting the current sample's deadbeat point:
    alpha <= 1 / |z^T (phi - gamma phi')|, where z^T(phi - gamma phi') is the per-sample
    interaction supplied as the curvature term. Maintained as a running minimum, starting
    at alpha = 1.
    """

    cfg: AlphaBoundConfig
    name: str = "alpha_bound"

    def init(self, parameters: PyTree) -> AlphaBoundState:
        alpha = jnp.float32(self.cfg.alpha_init)
        return AlphaBoundState(alpha=alpha)

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        gradient_trace = sum(
            jnp.sum(g * z)
            for g, z in zip(jax.tree.leaves(gradient), jax.tree.leaves(trace))
        )
        next_value, next_grad_trace = jax.jvp(bootstrap_fn, (params,), (trace,))
        curvature = gradient_trace - gamma * not_done * next_grad_trace
        return next_value, curvature

    def update(
        self,
        state: AlphaBoundState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        interaction: Array,
    ) -> tuple[PyTree, AlphaBoundState]:
        del gradient

        bound = 1.0 / jnp.maximum(jnp.abs(interaction), self.cfg.eps)
        alpha = jnp.minimum(state.alpha, bound)

        def compute_update(trace_leaf):
            return (alpha * td_error * trace_leaf).astype(self.cfg.dtype)

        updates = jax.tree.map(compute_update, trace)

        lox.log(
            {
                f"{self.name}/step_size": alpha.mean(),
                f"{self.name}/bound": bound.mean(),
            }
        )

        return updates, AlphaBoundState(alpha=alpha)
