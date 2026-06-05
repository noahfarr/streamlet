from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct
from streax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class ObGDConfig:
    lr: float
    kappa: float = 2.0
    beta2: float = 0.999
    eps: float = 1e-8
    adaptive: bool = struct.field(pytree_node=False, default=False)
    exact: bool = struct.field(pytree_node=False, default=False)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class ObGDState:
    second_moment: PyTree
    t_step: Array


@dataclass
class ObGD:

    cfg: ObGDConfig
    name: str = "obgd"

    def init(self, parameters: PyTree) -> ObGDState:
        second_moment = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=self.cfg.dtype),
            parameters,
        )
        return ObGDState(second_moment=second_moment, t_step=jnp.int32(0))

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        if not self.cfg.exact:
            return bootstrap_fn(params), None
        gradient_trace = sum(
            jnp.sum(g * z)
            for g, z in zip(jax.tree.leaves(gradient), jax.tree.leaves(trace))
        )
        next_value, next_grad_trace = jax.jvp(bootstrap_fn, (params,), (trace,))
        curvature = gradient_trace - gamma * not_done * next_grad_trace
        return next_value, curvature

    def update(
        self,
        state: ObGDState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array | None = None,
        squared_grad_norm: Array | None = None,
    ) -> tuple[PyTree, ObGDState]:
        del gradient, squared_grad_norm
        cfg = self.cfg
        next_t_step = state.t_step + 1

        if cfg.exact:
            if cfg.adaptive:
                raise ValueError(
                    "ObGD(exact=True) is not supported with adaptive=True: the "
                    "exact effective step size needs the preconditioned update "
                    "direction z/sqrt(v_hat), but v_hat is built from the in-update "
                    "(delta*z)^2 second moment and is unavailable when the curvature "
                    "term is formed. Use exact with non-adaptive ObGD."
                )
            if curvature is None:
                raise ValueError(
                    "ObGD(exact=True) requires the curvature term "
                    "z^T(grad_v(x) - gamma grad_v(x')); route this optimizer through "
                    "the algorithm's curvature branch, as Implicit/Calibrated are."
                )

        if cfg.adaptive:
            new_v = jax.tree.map(
                lambda v, t: (
                    cfg.beta2 * v + (1.0 - cfg.beta2) * jnp.square(td_error * t)
                ).astype(cfg.dtype),
                state.second_moment,
                trace,
            )
            v_hat = jax.tree.map(lambda v: v / (1.0 - cfg.beta2**next_t_step), new_v)
            scaled_trace_leaves = jax.tree.leaves(
                jax.tree.map(
                    lambda t, vh: jnp.abs(t) / jnp.sqrt(vh + cfg.eps),
                    trace,
                    v_hat,
                )
            )
            z_sum = sum(jnp.sum(leaf) for leaf in scaled_trace_leaves)
        else:
            new_v = state.second_moment
            v_hat = None
            z_sum = sum(jnp.sum(jnp.abs(leaf)) for leaf in jax.tree.leaves(trace))

        delta_bar = jnp.maximum(jnp.abs(td_error), 1.0)
        if cfg.exact:
            overshoot = jnp.abs(curvature)
        else:
            overshoot = delta_bar * z_sum
        step_size = cfg.lr / jnp.maximum(1.0, overshoot * cfg.lr * cfg.kappa)

        if cfg.adaptive:

            def compute_update(trace_leaf, v_hat_leaf):
                return (
                    step_size * td_error * trace_leaf / jnp.sqrt(v_hat_leaf + cfg.eps)
                )

            updates = jax.tree.map(compute_update, trace, v_hat)
        else:

            def compute_update(trace_leaf):
                return step_size * td_error * trace_leaf

            updates = jax.tree.map(compute_update, trace)

        updates = jax.tree.map(lambda u: u.astype(cfg.dtype), updates)

        lox.log(
            {
                f"{self.name}/step_size": step_size.mean(),
                f"{self.name}/z_sum": z_sum.mean(),
                f"{self.name}/delta_bar": delta_bar.mean(),
                f"{self.name}/overshoot": overshoot.mean(),
            }
        )

        return updates, ObGDState(second_moment=new_v, t_step=next_t_step)
