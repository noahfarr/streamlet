from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct

from streax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class AdaptiveConfig:
    gamma: float
    trace_lambda: float
    eta: float = 4.6e-4
    eps: float = 0.1
    clip: float = 1.0
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class AdaptiveState:
    second_moment: PyTree


@dataclass
class Adaptive:
    """Adaptive(λ) from "Revisiting Adam for Streaming RL" (arXiv:2605.06764).

    Maintains an EMA of the squared gradient (Adam-style) and uses the
    eligibility trace as the first-moment surrogate. The TD error is clipped to
    [-clip, clip] (default ±1, the derivative of the SmoothL1 loss).
    """

    cfg: AdaptiveConfig
    name: str = "adaptive"

    def init(self, parameters: PyTree) -> AdaptiveState:
        second_moment = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=self.cfg.dtype),
            parameters,
        )
        return AdaptiveState(second_moment=second_moment)

    def update(
        self,
        state: AdaptiveState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
    ) -> tuple[PyTree, AdaptiveState]:
        cfg = self.cfg
        gamma_lambda = cfg.gamma * cfg.trace_lambda

        new_v = jax.tree.map(
            lambda v, g: (
                gamma_lambda * v + (1.0 - gamma_lambda) * jnp.square(g)
            ).astype(cfg.dtype),
            state.second_moment,
            gradient,
        )

        clipped_delta = jnp.clip(td_error, -cfg.clip, cfg.clip)

        def compute_update(z, v):
            rho = z / (jnp.sqrt(v) + cfg.eps)
            return (cfg.eta * clipped_delta * rho).astype(cfg.dtype)

        updates = jax.tree.map(compute_update, trace, new_v)

        effective_lr = jax.tree.map(lambda v: cfg.eta / (jnp.sqrt(v) + cfg.eps), new_v)
        lr_leaves = jax.tree.leaves(effective_lr)
        step_size = sum(jnp.sum(x) for x in lr_leaves) / sum(x.size for x in lr_leaves)
        v_leaves = jax.tree.leaves(new_v)
        precond_rms = sum(jnp.sum(jnp.sqrt(x)) for x in v_leaves) / sum(
            x.size for x in v_leaves
        )

        lox.log(
            {
                f"{self.name}/step_size": step_size,
                f"{self.name}/precond_rms": precond_rms,
                f"{self.name}/clipped_delta": clipped_delta.mean(),
            }
        )

        return updates, AdaptiveState(second_moment=new_v)
