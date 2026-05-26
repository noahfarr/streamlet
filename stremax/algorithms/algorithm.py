from typing import Callable, Protocol, TypeVar

from stremax.utils.typing import Key

State = TypeVar("State")


class Algorithm(Protocol[State]):
    init: Callable[[Key], State]
    train: Callable[[Key, State, int], State]
    evaluate: Callable[[Key, State, int], State]
