import enum
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from streax.utils import broadcast
from streax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class CalibratedConfig:
    beta: float = 0.999
    eps: float = 1e-8
    nu: float = 1.0
    alpha_max: float = 1.0
    huber: bool = False
    huber_delta: float = 1.0
    precondition: bool = struct.field(pytree_node=False, default=False)
    beta2: float = 0.999
    precondition_eps: float = 0.1
    adaptive_clip: bool = struct.field(pytree_node=False, default=False)
    clip_multiplier: float = 20.0
    beta_clip: float = 0.999


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
    cfg: CalibratedConfig
    name: str = "calibrated"

    def init(self, parameters: PyTree, num_envs: int) -> CalibratedState:
        m_hat = s_hat = y_hat = jnp.zeros((num_envs,), dtype=jnp.float32)
        v = None
        if self.cfg.precondition:
            v = jax.tree.map(
                lambda p: jnp.zeros((num_envs, *p.shape), dtype=jnp.float32),
                parameters,
            )
        d_hat = None
        if self.cfg.adaptive_clip:
            d_hat = jnp.ones((num_envs,), dtype=jnp.float32)
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
            return jnp.where(has_grads, t / (jnp.sqrt(v_hat) + self.cfg.precondition_eps), t)

        return jax.tree.map(rescale, state.v, trace)

    def update(
        self,
        state: CalibratedState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        interaction: Array,
    ) -> tuple[PyTree, CalibratedState]:

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
            return jnp.sum(jnp.square(z_leaf), axis=tuple(range(1, z_leaf.ndim)))

        tree_norms = jax.tree.map(squared_norm, direction)
        squared_z_norm = jax.tree_util.tree_reduce(jnp.add, tree_norms)

        y_t = jnp.square(td_error) * squared_z_norm

        nu = self.cfg.nu

        alpha = jnp.maximum(0.0, state.m_hat) / (
            state.s_hat + nu * state.y_hat + self.cfg.eps
        )
        alpha = jnp.minimum(alpha, self.cfg.alpha_max)

        def compute_update(direction_leaf):
            return (
                broadcast(alpha, direction_leaf)
                * broadcast(td_error, direction_leaf)
                * direction_leaf
            ).mean(axis=0)

        updates = jax.tree.map(compute_update, direction)

        second_moment = jnp.square(interaction)

        m_hat = self.cfg.beta * state.m_hat + (1.0 - self.cfg.beta) * interaction
        s_hat = self.cfg.beta * state.s_hat + (1.0 - self.cfg.beta) * second_moment
        y_hat = self.cfg.beta * state.y_hat + (1.0 - self.cfg.beta) * y_t

        v = state.v
        if self.cfg.precondition:
            v = jax.tree.map(
                lambda v, g: self.cfg.beta2 * v + (1.0 - self.cfg.beta2) * jnp.square(g),
                state.v,
                gradient,
            )

        lox.log(
            {
                f"{self.name}/step_size": alpha.mean(),
                f"{self.name}/m_hat": m_hat.mean(),
                f"{self.name}/s_hat": s_hat.mean(),
                f"{self.name}/y_hat": y_hat.mean(),
                f"{self.name}/nu": jnp.mean(jnp.asarray(nu)),
                f"{self.name}/noise_ratio": (y_hat / (s_hat + self.cfg.eps)).mean(),
                f"{self.name}/rho": (nu * y_hat / (s_hat + self.cfg.eps)).mean(),
                f"{self.name}/expansive_fraction": (state.m_hat <= 0.0).mean(),
                f"{self.name}/cv2": (
                    s_hat / (jnp.square(m_hat) + self.cfg.eps) - 1.0
                ).mean(),
                f"{self.name}/update_norm": optax.global_norm(updates),
                f"{self.name}/absolute_td_error": jnp.abs(td_error).mean(),
            }
        )

        return updates, CalibratedState(
            m_hat=m_hat,
            s_hat=s_hat,
            y_hat=y_hat,
            v=v,
            d_hat=d_hat,
            step=step,
        )
