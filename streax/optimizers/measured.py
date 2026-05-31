import enum
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct

from streax.utils import broadcast
from streax.utils.typing import Array, PyTree


class MeasuredMode(enum.Enum):
    OPERATOR = "operator"
    FROBENIUS = "frobenius"


@struct.dataclass(frozen=True)
class MeasuredConfig:
    eta: float = 1.0
    beta: float = 0.999
    eps: float = 1e-8
    nu: float = 1.0
    alpha_max: float = 1.0
    huber: bool = False
    huber_delta: float = 1.0
    precondition: bool = struct.field(pytree_node=False, default=False)
    beta2: float = 0.999
    mode: MeasuredMode = struct.field(pytree_node=False, default=MeasuredMode.OPERATOR)
    adaptive_nu: bool = struct.field(pytree_node=False, default=False)


@struct.dataclass(frozen=True)
class MeasuredState:
    m_hat: Array
    s_hat: Array
    y_hat: Array
    v: PyTree = None
    t_hat: Array = None


@dataclass
class Measured:
    cfg: MeasuredConfig
    name: str = "measured"

    def init(self, parameters: PyTree, num_envs: int) -> MeasuredState:
        m_hat = s_hat = y_hat = jnp.zeros((num_envs,), dtype=jnp.float32)
        v = None
        if self.cfg.precondition:
            v = jax.tree.map(
                lambda p: jnp.ones((num_envs, *p.shape), dtype=jnp.float32),
                parameters,
            )
        t_hat = jnp.ones((num_envs,), dtype=jnp.float32) if self.cfg.adaptive_nu else None
        return MeasuredState(m_hat=m_hat, s_hat=s_hat, y_hat=y_hat, v=v, t_hat=t_hat)

    def precondition(self, state: MeasuredState, trace: PyTree) -> PyTree:
        if not self.cfg.precondition:
            return trace
        return jax.tree.map(
            lambda v, t: t / (jnp.sqrt(v) + self.cfg.eps), state.v, trace
        )

    def update(
        self,
        state: MeasuredState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        interaction: Array,
        squared_grad_norm: Array = None,
    ) -> tuple[PyTree, MeasuredState]:

        if self.cfg.huber:
            td_error = jnp.clip(td_error, -self.cfg.huber_delta, self.cfg.huber_delta)

        direction = self.precondition(state, trace)

        def squared_norm(z_leaf):
            return jnp.sum(jnp.square(z_leaf), axis=tuple(range(1, z_leaf.ndim)))

        tree_norms = jax.tree.map(squared_norm, direction)
        squared_z_norm = jax.tree_util.tree_reduce(jnp.add, tree_norms)

        y_t = jnp.square(td_error) * squared_z_norm

        if self.cfg.adaptive_nu:
            nu = 1.0 / (state.t_hat + self.cfg.eps)
        else:
            nu = self.cfg.nu

        alpha = (
            self.cfg.eta
            * jnp.maximum(0.0, state.m_hat)
            / (state.s_hat + nu * state.y_hat + self.cfg.eps)
        )
        alpha = jnp.minimum(alpha, self.cfg.alpha_max)

        def compute_update(direction_leaf):
            return (
                broadcast(alpha, direction_leaf)
                * broadcast(td_error, direction_leaf)
                * direction_leaf
            ).mean(axis=0)

        updates = jax.tree.map(compute_update, direction)

        if self.cfg.mode is MeasuredMode.FROBENIUS:
            second_moment = squared_grad_norm * squared_z_norm
        else:
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

        t_hat = state.t_hat
        if self.cfg.adaptive_nu:
            alpha_bar = jnp.maximum(0.0, state.m_hat) / (state.s_hat + self.cfg.eps)
            contraction = (
                1.0
                - 2.0 * alpha_bar * state.m_hat
                + jnp.square(alpha_bar) * state.s_hat
            )
            t_hat = jnp.maximum(
                state.t_hat * contraction + jnp.square(alpha_bar) * state.y_hat,
                self.cfg.eps,
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
            }
        )

        return updates, MeasuredState(
            m_hat=m_hat, s_hat=s_hat, y_hat=y_hat, v=v, t_hat=t_hat
        )
