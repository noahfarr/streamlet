import flax.linen as nn
import jax.numpy as jnp
from streamlet.utils.typing import Array


class L2Normalize(nn.Module):
    eps: float = 1e-12

    @nn.compact
    def __call__(self, x: Array) -> Array:
        return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), self.eps)
