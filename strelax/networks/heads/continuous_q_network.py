import flax.linen as nn
import jax.numpy as jnp
from strelax.utils.typing import Array


class ContinuousQNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, action: Array, **kwargs) -> Array:
        x = jnp.concatenate([x, action], axis=-1)
        q_value = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return jnp.squeeze(q_value, axis=-1)
