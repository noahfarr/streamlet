import distrax
import flax.linen as nn
import jax.numpy as jnp
from strelax.utils.typing import Array


class DiscreteQNetwork(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Array:
        q_values = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        return q_values


class VNetwork(nn.Module):
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> Array:
        v_value = nn.Dense(1, kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
        return v_value


class Categorical(nn.Module):
    action_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> distrax.Categorical:
        logits = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        return distrax.Categorical(logits=logits)
