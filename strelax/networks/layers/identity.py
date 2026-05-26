import flax.linen as nn
from strelax.utils.typing import Array


class Identity(nn.Module):

    @nn.compact
    def __call__(self, x: Array, *args, **kwargs) -> Array:
        return x
