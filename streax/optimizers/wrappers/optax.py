from dataclasses import dataclass

import jax
import lox
import optax
from flax import struct

from streax.utils import broadcast
from streax.utils.typing import Array, PyTree


@struct.dataclass(frozen=True)
class OptaxOptimizerState:
    opt_state: PyTree


@dataclass
class OptaxOptimizer:
    """Adapts any optax ``GradientTransformation`` to the streaming interface.

    The TD-scaled eligibility trace ``td_error * trace`` is the ascent
    direction; since the algorithm applies updates as ``params + updates``, we
    hand optax its negation (averaged over the environment axis) as the
    gradient to descend.
    """

    tx: optax.GradientTransformation
    name: str = "optimizer"

    def init(self, parameters: PyTree, num_envs: int) -> OptaxOptimizerState:
        return OptaxOptimizerState(opt_state=self.tx.init(parameters))

    def update(
        self,
        state: OptaxOptimizerState,
        gradient: PyTree,
        trace: PyTree | None = None,
        td_error: Array | None = None,
    ) -> tuple[PyTree, OptaxOptimizerState]:
        if trace is None:
            grad = gradient
        else:
            grad = jax.tree.map(
                lambda leaf: -(broadcast(td_error, leaf) * leaf).mean(axis=0), trace
            )
        updates, opt_state = self.tx.update(grad, state.opt_state)
        lox.log({f"{self.name}/update_norm": optax.global_norm(updates)})
        return updates, OptaxOptimizerState(opt_state=opt_state)
