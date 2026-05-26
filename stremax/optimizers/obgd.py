from dataclasses import dataclass

import jax
import jax.numpy as jnp
import lox
import optax
from flax import struct
from stremax.utils import broadcast
from stremax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class OBGDConfig:
    lr: float
    kappa: float = 2.0
    beta2: float = 0.999
    eps: float = 1e-8
    adaptive: bool = struct.field(pytree_node=False, default=False)


@struct.dataclass(frozen=True)
class OBGDState:
    second_moment: PyTree
    t_step: Array


@dataclass
class OBGD:

    cfg: OBGDConfig
    name: str = "obgd"

    def init(self, parameters: PyTree, num_envs: int) -> OBGDState:
        second_moment = jax.tree.map(
            lambda p: jnp.zeros((num_envs, *p.shape), dtype=jnp.float32),
            parameters,
        )
        return OBGDState(second_moment=second_moment, t_step=jnp.int32(0))

    def update(
        self,
        state: OBGDState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
    ) -> tuple[PyTree, OBGDState]:
        del gradient
        cfg = self.cfg
        next_t_step = state.t_step + 1

        if cfg.adaptive:
            new_v = jax.tree.map(
                lambda v, t: cfg.beta2 * v
                + (1.0 - cfg.beta2) * jnp.square(broadcast(td_error, t) * t),
                state.second_moment,
                trace,
            )
            v_hat = jax.tree.map(lambda v: v / (1.0 - cfg.beta2**next_t_step), new_v)
            scaled_trace_leaves = jax.tree.leaves(
                jax.tree.map(
                    lambda t, vh: jnp.abs(t) / (jnp.sqrt(vh) + cfg.eps),
                    trace,
                    v_hat,
                )
            )
            z_sum = sum(
                jnp.sum(leaf, axis=tuple(range(1, leaf.ndim)))
                for leaf in scaled_trace_leaves
            )
        else:
            new_v = state.second_moment
            v_hat = None
            z_sum = sum(
                jnp.sum(jnp.abs(leaf), axis=tuple(range(1, leaf.ndim)))
                for leaf in jax.tree.leaves(trace)
            )

        delta_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        step_size = cfg.lr / jnp.maximum(1.0, delta_bar * z_sum * cfg.lr * cfg.kappa)

        if cfg.adaptive:

            def compute_update(trace_leaf, v_hat_leaf):
                return (
                    broadcast(step_size, trace_leaf)
                    * broadcast(td_error, trace_leaf)
                    * trace_leaf
                    / (jnp.sqrt(v_hat_leaf) + cfg.eps)
                ).mean(axis=0)

            updates = jax.tree.map(compute_update, trace, v_hat)
        else:

            def compute_update(trace_leaf):
                return (
                    broadcast(step_size, trace_leaf)
                    * broadcast(td_error, trace_leaf)
                    * trace_leaf
                ).mean(axis=0)

            updates = jax.tree.map(compute_update, trace)

        lox.log(
            {
                f"{self.name}/step_size": step_size.mean(),
                f"{self.name}/z_sum": z_sum.mean(),
                f"{self.name}/delta_bar": delta_bar.mean(),
                f"{self.name}/update_norm": optax.global_norm(updates),
            }
        )

        return updates, OBGDState(second_moment=new_v, t_step=next_t_step)
