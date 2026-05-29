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
    s0: float = 1e-8
    alpha_max: float = 1.0
    kappa: float = 1.0


@struct.dataclass(frozen=True)
class MeasuredState:
    m_hat: Array
    s_hat: Array


@dataclass
class Measured:
    """Measured Updates (variance-optimal gradient descent).

    Sets the step size from running estimates of the first and second moments of the
    per-sample TD interaction X_t = (g_t - gamma * g'_t) . z_t, giving the
    variance-optimal counterpart to ObGD's overshoot bound:
    alpha = eta * max(E[X], 0) / E[X^2]. There is deliberately no base learning rate.
    The pre-update moments are used for the step, then X_t is folded in (causal
    ordering). No bias correction is stored: the 1 - beta^t factor cancels in
    the m_hat / s_hat ratio.

    The variance-optimal step is overshoot-free in expectation (E[alpha * X] =
    eta * E[X]^2 / E[X^2] <= eta <= 1), but not per-sample: a tail X_t or a stale
    moment estimate under non-stationarity can drive the realized contraction
    alpha * X_t past 1 and diverge. Since the TD error contracts as
    delta_new = delta * (1 - alpha * X_t), we additionally clamp the step by the
    exact per-sample curvature, alpha <= kappa / |X_t|, which guarantees
    |1 - alpha * X_t| <= 1 sample-by-sample (kappa = 1 => no overshoot). This uses
    the exact interaction X_t the optimizer already receives, so it is tighter
    than ObGD's ||z||_1 surrogate and needs no bound on the gradient norm. The
    clamp only binds on the tail events that would otherwise diverge; in the
    stationary, well-estimated regime it never fires and the variance-optimal
    step is recovered.

    The kappa clamp is one-sided: for X_t > 0 it caps the contraction at
    1 - alpha * X_t in [1 - kappa, 1], but for X_t <= 0 no positive step is
    contractive (1 - alpha * X_t = 1 + alpha * |X_t| >= 1), so the clamp alone
    would still permit up to kappa expansion. Since alpha >= 0 always, the
    max(m_hat, 0) gate only suppresses the step when the *average* interaction is
    non-positive; individual expansive samples slip through while the lagging EMA
    stays positive. We therefore gate the step to zero whenever X_t <= 0. Together
    with the kappa clamp this makes the realized contraction 1 - alpha * X_t lie
    in [0, 1] for *every* sample, upgrading the variance-optimal step's
    in-expectation non-overshoot (E[alpha * X] <= eta) to a per-sample guarantee.
    """

    cfg: MeasuredConfig
    name: str = "measured"

    def init(self, parameters: PyTree, num_envs: int) -> MeasuredState:
        del parameters
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        return MeasuredState(m_hat=zeros, s_hat=zeros)

    def update(
        self,
        state: MeasuredState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        interaction: Array,
    ) -> tuple[PyTree, MeasuredState]:
        del gradient
        cfg = self.cfg

        alpha = cfg.eta * jnp.maximum(state.m_hat, 0.0) / (state.s_hat + cfg.s0)
        alpha = jnp.minimum(alpha, cfg.alpha_max)
        alpha = jnp.minimum(alpha, cfg.kappa / (jnp.abs(interaction) + cfg.s0))
        alpha = jnp.where(interaction > 0.0, alpha, 0.0)

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
                f"{self.name}/contraction": jnp.abs(alpha * interaction).mean(),
                f"{self.name}/expansive_fraction": (interaction <= 0.0).mean(),
                f"{self.name}/update_norm": optax.global_norm(updates),
            }
        )

        return updates, MeasuredState(m_hat=m_hat, s_hat=s_hat)
