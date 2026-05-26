import distrax
import flax.linen as nn
import jax.numpy as jnp
from stremax.utils.typing import Array


class SquashedGaussian(nn.Module):
    action_dim: int
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros_init()

    @nn.compact
    def __call__(self, x: Array, **kwargs) -> distrax.Transformed:
        mean = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        log_std = nn.Dense(
            self.action_dim, kernel_init=self.kernel_init, bias_init=self.bias_init
        )(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        base = distrax.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        return distrax.Transformed(base, distrax.Block(distrax.Tanh(), ndims=1))
