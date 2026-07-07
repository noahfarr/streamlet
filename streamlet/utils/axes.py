import jax.numpy as jnp

from streamlet.utils.typing import Array


def ensure_axis(value: Array, size: int) -> Array:
    value = jnp.atleast_1d(jnp.asarray(value))
    _, *shape = value.shape
    return jnp.broadcast_to(value, (size, *shape))


def add_feature_axis(x: Array) -> Array:
    return jnp.expand_dims(x, axis=-1)


def remove_feature_axis(x: Array) -> Array:
    return jnp.squeeze(x, axis=-1)
