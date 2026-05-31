from typing import Callable

import distrax
import flax.linen as nn
import jax.numpy as jnp
from streax.utils.typing import Array


class StateDependentGaussian(nn.Module):
    action_dim: int
    transform: Callable[[Array], Array] = jnp.exp
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> distrax.MultivariateNormalDiag:
        mean = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        pre_std = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        std = self.transform(pre_std)
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
