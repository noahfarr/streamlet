from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from stremax.utils import broadcast
from stremax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class MeasuredConfig:
    eta: float = 1.0
    beta: float = 0.999
    eps: float = 1e-8
    nu: float = 0.01


@struct.dataclass(frozen=True)
class MeasuredState:
    m_hat: Array
    s_hat: Array
    y_hat: Array


@dataclass
class Measured:
    cfg: MeasuredConfig
    name: str = "measured"

    def init(self, parameters: PyTree, num_envs: int) -> MeasuredState:
        m_hat = s_hat = y_hat = jnp.zeros((num_envs,), dtype=jnp.float32)
        return MeasuredState(m_hat=m_hat, s_hat=s_hat, y_hat=y_hat)

    def update(
        self,
        state: MeasuredState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        interaction: Array,
    ) -> tuple[PyTree, MeasuredState]:

        def squared_norm(z_leaf):
            return jnp.sum(jnp.square(z_leaf), axis=tuple(range(1, z_leaf.ndim)))

        tree_norms = jax.tree.map(squared_norm, trace)
        squared_z_norm = jax.tree_util.tree_reduce(jnp.add, tree_norms)

        y_t = jnp.square(td_error) * squared_z_norm

        alpha = (
            self.cfg.eta
            * jnp.maximum(0.0, state.m_hat)
            / (state.s_hat + self.cfg.nu * state.y_hat + self.cfg.eps)
        )
        alpha = jnp.minimum(alpha, 1.0)

        def compute_update(trace_leaf):
            return (
                broadcast(alpha, trace_leaf)
                * broadcast(td_error, trace_leaf)
                * trace_leaf
            ).mean(axis=0)

        updates = jax.tree.map(compute_update, trace)

        m_hat = self.cfg.beta * state.m_hat + (1.0 - self.cfg.beta) * interaction
        s_hat = self.cfg.beta * state.s_hat + (1.0 - self.cfg.beta) * jnp.square(
            interaction
        )
        y_hat = self.cfg.beta * state.y_hat + (1.0 - self.cfg.beta) * y_t

        lox.log(
            {
                f"{self.name}/step_size": alpha.mean(),
                f"{self.name}/m_hat": m_hat.mean(),
                f"{self.name}/s_hat": s_hat.mean(),
                f"{self.name}/y_hat": y_hat.mean(),
                f"{self.name}/expansive_fraction": (state.m_hat <= 0.0).mean(),
                f"{self.name}/cv2": (
                    s_hat / (jnp.square(m_hat) + self.cfg.eps) - 1.0
                ).mean(),
                f"{self.name}/update_norm": optax.global_norm(updates),
            }
        )

        return updates, MeasuredState(m_hat=m_hat, s_hat=s_hat, y_hat=y_hat)
