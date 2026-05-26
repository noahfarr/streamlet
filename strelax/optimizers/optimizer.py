from typing import Protocol, runtime_checkable

from strelax.utils.typing import Array, PyTree


@runtime_checkable
class Optimizer(Protocol):
    def init(self, parameters: PyTree, num_envs: int) -> PyTree: ...

    def update(
        self,
        state: PyTree,
        gradient: PyTree,
        trace: PyTree,
        td_error: Array,
    ) -> tuple[PyTree, PyTree]: ...
