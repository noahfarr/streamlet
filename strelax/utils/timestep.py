from flax import struct

from strelax.utils.typing import Array


@struct.dataclass(frozen=True)
class Timestep:
    obs: Array | None = None
    action: Array | None = None
    reward: Array | None = None
    done: Array | None = None
