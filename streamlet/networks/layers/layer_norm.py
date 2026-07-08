from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from streamlet.utils.typing import Array


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def layer_norm(x: Array, scale: Array, bias: Array, eps: float) -> Array:
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    x_hat = centered * jax.lax.rsqrt(var + eps)
    return x_hat * scale + bias


@layer_norm.defjvp
def layer_norm_jvp(eps: float, primals, tangents):
    x, scale, bias = primals
    dx, dscale, dbias = tangents

    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)
    x_hat = centered * rstd
    y = x_hat * scale + bias

    mean_dx = dx.mean(axis=-1, keepdims=True)
    mean_dx_xhat = jnp.mean(dx * x_hat, axis=-1, keepdims=True)
    dx_hat = rstd * (dx - mean_dx - x_hat * mean_dx_xhat)
    dy = dx_hat * scale + x_hat * dscale + dbias
    return y, dy


class LayerNorm(nn.Module):
    epsilon: float = 1e-6
    use_scale: bool = True
    use_bias: bool = True
    scale_init: Callable = nn.initializers.ones
    bias_init: Callable = nn.initializers.zeros
    dtype: Any = None

    @nn.compact
    def __call__(self, x: Array) -> Array:
        features = x.shape[-1]
        dtype = self.dtype or x.dtype
        if self.use_scale:
            scale = self.param("scale", self.scale_init, (features,), dtype)
        else:
            scale = jnp.ones((features,), dtype)
        if self.use_bias:
            bias = self.param("bias", self.bias_init, (features,), dtype)
        else:
            bias = jnp.zeros((features,), dtype)
        return layer_norm(x, scale, bias, self.epsilon)
