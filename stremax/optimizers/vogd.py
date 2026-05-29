from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from stremax.utils import broadcast
from stremax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class VOGDConfig:
    eta: float = 1.0
    beta: float = 0.999
    s0: float = 1e-8
    alpha_max: float = 1.0


@struct.dataclass(frozen=True)
class VOGDState:
    m_hat: Array
    s_hat: Array


@dataclass
class VOGD:
    """VOGD = Variance-Optimal Gradient Descent.

    Sets the step size from running estimates of the first and second moments of the
    per-sample TD interaction X_t = (g_t - gamma * g'_t) . z_t, giving the
    variance-optimal counterpart to ObGD's overshoot bound:
    alpha = eta * max(E[X], 0) / E[X^2]. There is deliberately no base learning rate.
    The pre-update moments are used for the step, then X_t is folded in (causal
    ordering). No bias correction is stored: the 1 - beta^t factor cancels in
    the m_hat / s_hat ratio.
    """

    cfg: VOGDConfig
    name: str = "vogd"

    def init(self, parameters: PyTree, num_envs: int) -> VOGDState:
        del parameters
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        return VOGDState(m_hat=zeros, s_hat=zeros)

    def update(
        self,
        state: VOGDState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        interaction: Array,
    ) -> tuple[PyTree, VOGDState]:
        del gradient
        cfg = self.cfg

        alpha = cfg.eta * jnp.maximum(state.m_hat, 0.0) / (state.s_hat + cfg.s0)
        alpha = jnp.minimum(alpha, cfg.alpha_max)

        def compute_update(trace_leaf):
            return (
                broadcast(alpha, trace_leaf)
                * broadcast(td_error, trace_leaf)
                * trace_leaf
            ).mean(axis=0)

        updates = jax.tree.map(compute_update, trace)

        m_hat = cfg.beta * state.m_hat + (1.0 - cfg.beta) * interaction
        s_hat = cfg.beta * state.s_hat + (1.0 - cfg.beta) * jnp.square(interaction)

        lox.log(
            {
                f"{self.name}/step_size": alpha.mean(),
                f"{self.name}/m_hat": m_hat.mean(),
                f"{self.name}/s_hat": s_hat.mean(),
                f"{self.name}/update_norm": optax.global_norm(updates),
            }
        )

        return updates, VOGDState(m_hat=m_hat, s_hat=s_hat)
