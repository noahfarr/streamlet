from flax import struct

from streax.utils.timestep import Timestep
from streax.utils.typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Timestep | None = None
    second: Timestep | None = None
    aux: PyTree | None = None
