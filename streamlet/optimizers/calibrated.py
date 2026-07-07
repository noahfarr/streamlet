import enum
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct

from streamlet.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class CalibratedConfig:
    beta: float = 0.999
    eps: float = 1e-8
    nu: float = 1.0
    adaptive_nu: bool = struct.field(pytree_node=False, default=False)
    rho_target: float = 1.0
    alpha_max: float = 1.0
    huber: bool = False
    huber_delta: float = 1.0
    precondition: bool = struct.field(pytree_node=False, default=False)
    beta2: float = 0.999
    precondition_eps: float = 0.1
    adaptive_clip: bool = struct.field(pytree_node=False, default=False)
    clip_multiplier: float = 20.0
    beta_clip: float = 0.999
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class CalibratedState:
    m_hat: Array
    s_hat: Array
    y_hat: Array
    v: PyTree = None
    d_hat: Array = None
    step: Array = None


@dataclass
class Calibrated:
    """Calibrated (measured) step size for the critic of a streaming agent.

    A single per-step scalar quadratic in the step size ``alpha`` minimizes the
    next value-error second moment. With ``X`` the interaction (contraction-rate
    sample, the bootstrap curvature) and ``y = delta^2 ||z||^2`` the
    target-variance sample::

        alpha = max(0, E[X]) / (E[X^2] + nu * E[y])

    The ``max(0, .)`` is the expansiveness off-switch: a non-positive ``E[X]``
    means no positive step contracts (the semi-gradient operator is asymmetric)
    and the head freezes. The update uses the preconditioned direction ``z``, the
    noise functional ``E[delta^2 ||z||^2]``, the update form ``alpha * delta * z``,
    and the optional Huber / adaptive TD-error clipping. The caller must supply the
    bootstrap curvature ``interaction``; the no-curvature (measured-policy-gradient)
    path has been removed.
    """

    cfg: CalibratedConfig
    name: str = "calibrated"

    def init(self, parameters: PyTree) -> CalibratedState:
        m_hat = s_hat = y_hat = jnp.float32(0.0)
        v = None
        if self.cfg.precondition:
            v = jax.tree.map(
                lambda p: jnp.zeros(p.shape, dtype=self.cfg.dtype),
                parameters,
            )
        d_hat = None
        if self.cfg.adaptive_clip:
            d_hat = jnp.float32(1.0)
        step = jnp.zeros((), dtype=jnp.float32)
        return CalibratedState(
            m_hat=m_hat,
            s_hat=s_hat,
            y_hat=y_hat,
            v=v,
            d_hat=d_hat,
            step=step,
        )

    def precondition(self, state: CalibratedState, trace: PyTree) -> PyTree:
        if not self.cfg.precondition:
            return trace
        has_grads = state.step > 0
        bias = jnp.where(has_grads, 1.0 - self.cfg.beta2**state.step, 1.0)

        def rescale(v, t):
            v_hat = v / bias
            return jnp.where(
                has_grads, t / (jnp.sqrt(v_hat) + self.cfg.precondition_eps), t
            )

        return jax.tree.map(rescale, state.v, trace)

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        interaction_trace = self.precondition(state, trace)
        gradient_trace = sum(
            jnp.sum(g * z)
            for g, z in zip(
                jax.tree.leaves(gradient), jax.tree.leaves(interaction_trace)
            )
        )
        next_value, next_grad_trace = jax.jvp(
            bootstrap_fn, (params,), (interaction_trace,)
        )
        curvature = gradient_trace - gamma * not_done * next_grad_trace
        return next_value, curvature

    def update(
        self,
        state: CalibratedState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        interaction: Array | None = None,
    ) -> tuple[PyTree, CalibratedState]:
        if interaction is None:
            raise ValueError(
                "Calibrated requires the bootstrap curvature `interaction` "
                "(critic mode); the no-curvature actor path has been removed. "
                "Route this optimizer through the algorithm's curvature branch."
            )

        step = state.step + 1.0
        d_hat = state.d_hat
        if self.cfg.adaptive_clip:
            d_hat = self.cfg.beta_clip * state.d_hat + (1.0 - self.cfg.beta_clip) * (
                td_error * td_error
            )
            ceiling = self.cfg.clip_multiplier * jnp.sqrt(
                d_hat / (1.0 - self.cfg.beta_clip**step)
            )
            td_error = jnp.sign(td_error) * jnp.minimum(jnp.abs(td_error), ceiling)
        elif self.cfg.huber:
            td_error = jnp.clip(td_error, -self.cfg.huber_delta, self.cfg.huber_delta)

        direction = self.precondition(state, trace)

        def squared_norm(z_leaf):
            return jnp.sum(jnp.square(z_leaf))

        tree_norms = jax.tree.map(squared_norm, direction)
        squared_z_norm = jax.tree_util.tree_reduce(jnp.add, tree_norms)

        # Streaming sample g_hat = delta * z (the update direction, pre-step).
        g_hat = jax.tree.map(lambda z: td_error * z, direction)
        y_t = jnp.square(td_error) * squared_z_norm

        # Measured critic step: alpha = max(0, E[X]) / (E[X^2] + nu E[y]). The gain
        # is computed from the PRIOR (pre-update) moments, so it is F_{t-1}-
        # measurable -- independent of the current sample it multiplies. That keeps
        # alpha * delta * z an unbiased descent step (the fresh-moment alternative
        # correlates the gain with its own direction) and gives a free warmup: the
        # zero-initialized moments yield alpha = 0 until real statistics fill in.
        if self.cfg.adaptive_nu:
            nu = self.cfg.rho_target * state.s_hat / (state.y_hat + self.cfg.eps)
        else:
            nu = self.cfg.nu
        numerator = jnp.maximum(0.0, state.m_hat)
        denominator = state.s_hat + nu * state.y_hat

        alpha = numerator / (denominator + self.cfg.eps)
        alpha = jnp.minimum(alpha, self.cfg.alpha_max)

        updates = jax.tree.map(lambda g: (alpha * g).astype(self.cfg.dtype), g_hat)

        # Fold the current sample into the moment estimates for the next step.
        second_moment = jnp.square(interaction)
        m_hat = self.cfg.beta * state.m_hat + (1.0 - self.cfg.beta) * interaction
        s_hat = self.cfg.beta * state.s_hat + (1.0 - self.cfg.beta) * second_moment
        y_hat = self.cfg.beta * state.y_hat + (1.0 - self.cfg.beta) * y_t

        v = state.v
        if self.cfg.precondition:
            v = jax.tree.map(
                lambda v, g: (
                    self.cfg.beta2 * v + (1.0 - self.cfg.beta2) * jnp.square(g)
                ).astype(self.cfg.dtype),
                state.v,
                gradient,
            )

        log_dict = {
            f"{self.name}/step_size": alpha.mean(),
            f"{self.name}/y_hat": y_hat.mean(),
            f"{self.name}/absolute_td_error": jnp.abs(td_error).mean(),
            f"{self.name}/m_hat": m_hat.mean(),
            f"{self.name}/s_hat": s_hat.mean(),
            f"{self.name}/nu": jnp.mean(jnp.asarray(nu)),
            f"{self.name}/noise_ratio": (y_hat / (s_hat + self.cfg.eps)).mean(),
            f"{self.name}/rho": (nu * y_hat / (s_hat + self.cfg.eps)).mean(),
            f"{self.name}/expansive_fraction": (state.m_hat <= 0.0).mean(),
            f"{self.name}/cv2": (
                s_hat / (jnp.square(m_hat) + self.cfg.eps) - 1.0
            ).mean(),
        }
        lox.log(log_dict)

        return updates, CalibratedState(
            m_hat=m_hat,
            s_hat=s_hat,
            y_hat=y_hat,
            v=v,
            d_hat=d_hat,
            step=step,
        )
