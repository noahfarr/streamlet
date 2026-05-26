from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from stremax.utils import broadcast
from stremax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class AdaptiveQConfig:
    gamma: float
    trace_lambda: float
    eta: float = 4.6e-4
    eps: float = 0.1
    clip: float = 1.0


@struct.dataclass(frozen=True)
class AdaptiveQState:
    second_moment: PyTree


@dataclass
class AdaptiveQ:
    """Adaptive Q(λ) from "Revisiting Adam for Streaming RL" (arXiv:2605.06764).

    Maintains an EMA of the squared gradient (Adam-style) and uses the
    eligibility trace as the first-moment surrogate. The TD error is clipped to
    [-clip, clip] (default ±1, the derivative of the SmoothL1 loss).
    """

    cfg: AdaptiveQConfig
    name: str = "adaptive_q"

    def init(self, parameters: PyTree, num_envs: int) -> AdaptiveQState:
        second_moment = jax.tree.map(
            lambda p: jnp.zeros((num_envs, *p.shape), dtype=jnp.float32),
            parameters,
        )
        return AdaptiveQState(second_moment=second_moment)

    def update(
        self,
        state: AdaptiveQState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
    ) -> tuple[PyTree, AdaptiveQState]:
        cfg = self.cfg
        gamma_lambda = cfg.gamma * cfg.trace_lambda

        new_v = jax.tree.map(
            lambda v, g: gamma_lambda * v + (1.0 - gamma_lambda) * jnp.square(g),
            state.second_moment,
            gradient,
        )

        clipped_delta = jnp.clip(td_error, -cfg.clip, cfg.clip)

        def compute_update(z, v):
            rho = z / (jnp.sqrt(v) + cfg.eps)
            return (cfg.eta * broadcast(clipped_delta, rho) * rho).mean(axis=0)

        updates = jax.tree.map(compute_update, trace, new_v)

        lox.log(
            {
                f"{self.name}/update_norm": optax.global_norm(updates),
                f"{self.name}/clipped_delta": clipped_delta.mean(),
            }
        )

        return updates, AdaptiveQState(second_moment=new_v)
