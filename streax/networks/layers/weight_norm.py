from typing import Any, Callable

import flax.linen as nn
import jax.numpy as jnp

from streax.utils.typing import Array

default_kernel_init = nn.initializers.lecun_normal()


class WeightNorm(nn.Module):
    features: int
    use_scale: bool = False
    use_bias: bool = False
    scale_init: Callable = nn.initializers.ones
    bias_init: Callable = nn.initializers.zeros
    kernel_init: Callable = default_kernel_init
    epsilon: float = 1e-8
    dtype: Any = None

    @nn.compact
    def __call__(self, x: Array) -> Array:
        dtype = self.dtype or x.dtype
        kernel = self.param("kernel", self.kernel_init, (x.shape[-1], self.features))
        kernel = kernel / (jnp.sqrt(jnp.sum(kernel**2)) + self.epsilon)
        y = x @ kernel
        if self.use_scale:
            y = y * self.param("scale", self.scale_init, (self.features,), dtype)
        if self.use_bias:
            y = y + self.param("bias", self.bias_init, (self.features,), dtype)
        return y
