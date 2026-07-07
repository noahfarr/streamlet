from typing import Callable, Protocol, TypeVar

from streamlet.utils.typing import Array, PyTree

State = TypeVar("State")


class Optimizer(Protocol[State]):
    init: Callable[[PyTree], State]
    update: Callable[..., tuple[PyTree, State]]

    def bootstrap(
        self,
        state: State,
        params: PyTree,
        gradient: PyTree,
        trace: PyTree,
        bootstrap_fn: Callable[[PyTree], Array],
        gamma: float,
        not_done: Array,
    ) -> tuple[Array, Array | None]: ...
