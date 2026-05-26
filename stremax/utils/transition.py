from flax import struct

from stremax.utils.timestep import Timestep
from stremax.utils.typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Timestep | None = None
    second: Timestep | None = None
    aux: PyTree | None = None
