from typing import Callable, Protocol, TypeVar

from streax.utils.typing import PyTree

State = TypeVar("State")


class Optimizer(Protocol[State]):
    init: Callable[[PyTree], State]
    update: Callable[..., tuple[PyTree, State]]
