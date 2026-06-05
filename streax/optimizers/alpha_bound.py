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

    def init(self, parameters: PyTree, num_envs: int) -> AlphaBoundState:
        alpha = jnp.full((num_envs,), self.cfg.alpha_init, dtype=jnp.float32)
        return AlphaBoundState(alpha=alpha)

    def precondition(self, state: AlphaBoundState, trace: PyTree) -> PyTree:
        return trace

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
            return (
                (
                    broadcast(alpha, trace_leaf)
                    * broadcast(td_error, trace_leaf)
                    * trace_leaf
                )
                .mean(axis=0)
                .astype(self.cfg.dtype)
            )

        updates = jax.tree.map(compute_update, trace)

        lox.log(
            {
                f"{self.name}/step_size": alpha.mean(),
                f"{self.name}/bound": bound.mean(),
            }
        )

        return updates, AlphaBoundState(alpha=alpha)
