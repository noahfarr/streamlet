import flax.linen as nn
from strelax.utils.typing import Array


class VNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Array:
        v_value = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return v_value
