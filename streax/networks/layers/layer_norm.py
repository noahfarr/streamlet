from functools import partial
from typing import Any, Callable

import flax.linen as nn
import jax
import jax.numpy as jnp

from streax.utils.typing import Array


@partial(jax.custom_vjp, nondiff_argnums=(3,))
def _layer_norm(x: Array, scale: Array, bias: Array, eps: float) -> Array:
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    x_hat = centered * jax.lax.rsqrt(var + eps)
    return x_hat * scale + bias


def _layer_norm_fwd(x: Array, scale: Array, bias: Array, eps: float):
    mean = x.mean(axis=-1, keepdims=True)
    centered = x - mean
    var = jnp.mean(jnp.square(centered), axis=-1, keepdims=True)
    rstd = jax.lax.rsqrt(var + eps)
    x_hat = centered * rstd
    y = x_hat * scale + bias
    return y, (x_hat, rstd, scale)


def _layer_norm_bwd(eps: float, res, g: Array):
    del eps
    x_hat, rstd, scale = res
    reduce_axes = tuple(range(g.ndim - 1))

    g_bias = g.sum(axis=reduce_axes)
    g_scale = jnp.sum(g * x_hat, axis=reduce_axes)

    gy = g * scale
    mean_gy = gy.mean(axis=-1, keepdims=True)
    mean_gy_xhat = jnp.mean(gy * x_hat, axis=-1, keepdims=True)
    g_x = rstd * (gy - mean_gy - x_hat * mean_gy_xhat)

    return g_x, g_scale, g_bias


_layer_norm.defvjp(_layer_norm_fwd, _layer_norm_bwd)


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
        return _layer_norm(x, scale, bias, self.epsilon)
