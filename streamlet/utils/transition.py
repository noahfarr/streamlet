from flax import struct

from streamlet.utils.timestep import Timestep
from streamlet.utils.typing import PyTree


@struct.dataclass(frozen=True)
class Transition:
    first: Timestep | None = None
    second: Timestep | None = None
    aux: PyTree | None = None
