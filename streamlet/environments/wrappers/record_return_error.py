from typing import Any

import jax.numpy as jnp
import lox
from flax import struct
from gymnax.environments import environment

from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class RecordReturnErrorState:
    env_state: environment.EnvState
    predictions: Array
    cumulants: Array
    count: int

    def __getattr__(self, name):
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class RecordReturnError:
    def __init__(self, env, gamma: float, horizon: int):
        self._env = env
        self._gamma = gamma
        self._horizon = horizon
        self._weights = gamma ** jnp.arange(horizon, dtype=jnp.float32)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, RecordReturnErrorState]:
        obs, env_state = self._env.reset(key, params)
        zeros = jnp.zeros((self._horizon,), dtype=jnp.float32)
        return obs, RecordReturnErrorState(env_state, zeros, zeros, 0)

    def step(
        self,
        key: Key,
        state: RecordReturnErrorState,
        action: Array,
        params: EnvParams | None = None,
    ) -> tuple[Array, RecordReturnErrorState, Array, bool, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        index = state.count % self._horizon
        predictions = state.predictions.at[index].set(jnp.asarray(action, jnp.float32))
        cumulants = state.cumulants.at[index].set(jnp.asarray(reward, jnp.float32))
        oldest = (state.count + 1) % self._horizon
        order = (oldest + jnp.arange(self._horizon)) % self._horizon
        truncated_return = jnp.sum(self._weights * cumulants[order])
        error = jnp.square(predictions[oldest] - truncated_return)
        lox.log(
            {
                "return_error": jnp.where(
                    state.count + 1 >= self._horizon, error, jnp.nan
                )
            }
        )
        state = RecordReturnErrorState(env_state, predictions, cumulants, state.count + 1)
        return obs, state, reward, done, info
