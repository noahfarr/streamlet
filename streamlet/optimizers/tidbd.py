from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import lox
from flax import struct

from streamlet.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class TIDBDConfig:
    alpha_init: float = 1e-3
    theta: float = 0.01
    kappa: float | None = None
    beta_min: float = -20.0
    beta_max: float = 2.0
    adapt_paths: tuple[str, ...] | None = struct.field(pytree_node=False, default=None)
    dtype: Any = struct.field(pytree_node=False, default=jnp.float32)


@struct.dataclass(frozen=True)
class TIDBDState:
    beta: PyTree
    h: PyTree


@dataclass
class TIDBD:

    cfg: TIDBDConfig
    name: str = "tidbd"

    def init(self, parameters: PyTree) -> TIDBDState:
        beta_init = jnp.log(jnp.asarray(self.cfg.alpha_init, dtype=self.cfg.dtype))
        beta = jax.tree.map(
            lambda p: jnp.full(p.shape, beta_init, dtype=self.cfg.dtype), parameters
        )
        h = jax.tree.map(
            lambda p: jnp.zeros(p.shape, dtype=self.cfg.dtype), parameters
        )
        return TIDBDState(beta=beta, h=h)

    def theta_for(self, path) -> float:
        if self.cfg.adapt_paths is None:
            return self.cfg.theta
        joined = "/".join(str(getattr(key, "key", key)) for key in path)
        if any(name in joined for name in self.cfg.adapt_paths):
            return self.cfg.theta
        return 0.0

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        del state, gradient, trace, gamma, not_done
        return bootstrap_fn(params), None

    def update(
        self,
        state: TIDBDState,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
        curvature: Array | None = None,
    ) -> tuple[PyTree, TIDBDState]:
        del curvature
        cfg = self.cfg

        beta = jax.tree_util.tree_map_with_path(
            lambda path, b, g, h: jnp.clip(
                b + self.theta_for(path) * td_error * g * h, cfg.beta_min, cfg.beta_max
            ).astype(cfg.dtype),
            state.beta,
            gradient,
            state.h,
        )
        alpha = jax.tree.map(jnp.exp, beta)

        if cfg.kappa is not None:
            scaled = sum(
                jnp.sum(a * jnp.abs(z))
                for a, z in zip(jax.tree.leaves(alpha), jax.tree.leaves(trace))
            )
            delta_bar = jnp.maximum(jnp.abs(td_error), 1.0)
            overshoot = delta_bar * scaled * cfg.kappa
            bound = 1.0 / jnp.maximum(1.0, overshoot)
            alpha = jax.tree.map(lambda a: a * bound, alpha)
        else:
            bound = jnp.float32(1.0)

        updates = jax.tree.map(
            lambda a, z: (a * td_error * z).astype(cfg.dtype), alpha, trace
        )

        h = jax.tree.map(
            lambda h_leaf, a, g, z, u: (
                h_leaf * jnp.clip(1.0 - a * g * z, 0.0, 1.0) + u
            ).astype(cfg.dtype),
            state.h,
            alpha,
            gradient,
            trace,
            updates,
        )

        alpha_leaves = jax.tree.leaves(alpha)
        alpha_mean = sum(jnp.sum(a) for a in alpha_leaves) / sum(
            a.size for a in alpha_leaves
        )
        alpha_max = jnp.max(jnp.stack([jnp.max(a) for a in alpha_leaves]))

        lox.log(
            {
                f"{self.name}/step_size": alpha_mean,
                f"{self.name}/max_step_size": alpha_max,
                f"{self.name}/bound": jnp.mean(bound),
            }
        )

        return updates, TIDBDState(beta=beta, h=h)
