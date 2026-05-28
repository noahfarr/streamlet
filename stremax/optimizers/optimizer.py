from typing import Callable, Protocol, TypeVar

from stremax.utils.typing import PyTree

State = TypeVar("State")


class Optimizer(Protocol[State]):
    init: Callable[[PyTree, int], State]
    update: Callable[..., tuple[PyTree, State]]
