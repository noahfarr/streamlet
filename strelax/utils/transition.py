from flax import struct

from strelax.utils.timestep import Timestep
from strelax.utils.typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Timestep | None = None
    second: Timestep | None = None
    aux: PyTree | None = None
