from flax import struct

from streax.utils.typing import Array


@struct.dataclass(frozen=True)
class Timestep:
    obs: Array | None = None
    action: Array | None = None
    reward: Array | None = None
    done: Array | None = None

    def __iter__(self):
        return iter((self.obs, self.action, self.reward, self.done))
