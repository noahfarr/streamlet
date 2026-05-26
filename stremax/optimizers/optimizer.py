from typing import Callable, Protocol, TypeVar

from stremax.utils.typing import Array, PyTree

State = TypeVar("State")


class Optimizer(Protocol[State]):
    init: Callable[[PyTree, int], State]
    update: Callable[[State, PyTree, PyTree, Array], tuple[PyTree, State]]
