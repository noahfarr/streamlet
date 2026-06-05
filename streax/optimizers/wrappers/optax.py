from dataclasses import dataclass

import jax
import lox
import optax
from flax import struct

from streax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class OptaxOptimizerState:
    opt_state: PyTree


@dataclass
class OptaxOptimizer:

    tx: optax.GradientTransformation
    name: str = "optimizer"

    def init(self, parameters: PyTree) -> OptaxOptimizerState:
        return OptaxOptimizerState(opt_state=self.tx.init(parameters))

    def bootstrap(self, state, params, gradient, trace, bootstrap_fn, gamma, not_done):
        return bootstrap_fn(params), None

    def update(
        self,
        state: OptaxOptimizerState,
        gradient: PyTree,
        trace: PyTree | None = None,
        td_error: Array | None = None,
        curvature: Array | None = None,
    ) -> tuple[PyTree, OptaxOptimizerState]:
        if trace is None:
            grad = gradient
        else:
            grad = jax.tree.map(lambda leaf: -(td_error * leaf), trace)
        updates, opt_state = self.tx.update(grad, state.opt_state)
        return updates, OptaxOptimizerState(opt_state=opt_state)
