from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import GymnaxWrapper

from streamlet.utils import canonicalize_dtype
from streamlet.utils.typing import Array, Key


def _space_tree(space: spaces.Space) -> Any:
    return (
        {key: _space_tree(value) for key, value in space.spaces.items()}
        if isinstance(space, spaces.Dict)
        else tuple(_space_tree(value) for value in space.spaces)
        if isinstance(space, spaces.Tuple)
        else space
    )


def _leaf_spaces(space: spaces.Space) -> list[spaces.Space]:
    return jax.tree.leaves(
        _space_tree(space), is_leaf=lambda leaf: isinstance(leaf, spaces.Space)
    )


def _bounds(space: spaces.Space) -> tuple[Array, Array]:
    if isinstance(space, spaces.Discrete):
        low, high = 0, space.n - 1
    elif isinstance(space, spaces.Box):
        low, high = space.low, space.high
    else:
        raise NotImplementedError(
            f"Cannot flatten a {type(space).__name__} observation space"
        )
    return (
        jnp.broadcast_to(jnp.asarray(low), space.shape).ravel(),
        jnp.broadcast_to(jnp.asarray(high), space.shape).ravel(),
    )


class FlattenObservationWrapper(GymnaxWrapper):
    """Concatenate a pytree observation into a single vector.

    Leaves are raveled and joined in the order `jax.tree` flattens them, which
    sorts dictionary keys, so the layout stays stable across resets and steps.
    """

    def __init__(self, env, dtype: Any | None = None):
        super().__init__(env)
        self.dtype = None if dtype is None else canonicalize_dtype(dtype)

    def _flatten(self, observation: Any) -> Array:
        leaves = [jnp.ravel(leaf) for leaf in jax.tree.leaves(observation)]
        if self.dtype is not None:
            leaves = [leaf.astype(self.dtype) for leaf in leaves]
        return jnp.concatenate(leaves)

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, environment.EnvState]:
        observation, env_state = self._env.reset(key, params)
        return self._flatten(observation), env_state

    def step(
        self,
        key: Key,
        state: environment.EnvState,
        action: int | float,
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, environment.EnvState, float, bool, dict]:
        observation, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return self._flatten(observation), env_state, reward, done, info

    def observation_space(
        self, params: environment.EnvParams | None = None
    ) -> spaces.Box:
        leaves = _leaf_spaces(self._env.observation_space(params))
        lows, highs = zip(*(_bounds(leaf) for leaf in leaves))
        low = jnp.concatenate(lows)
        dtype = self.dtype or jnp.result_type(*(leaf.dtype for leaf in leaves))
        return spaces.Box(
            low=low, high=jnp.concatenate(highs), shape=low.shape, dtype=dtype
        )
