import jax.numpy as jnp

from streax.utils.typing import Array


def ensure_axis(value: Array, size: int) -> Array:
    value = jnp.atleast_1d(jnp.asarray(value))
    _, *shape = value.shape
    return jnp.broadcast_to(value, (size, *shape))
