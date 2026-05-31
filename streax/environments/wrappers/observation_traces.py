from typing import Union

import jax.numpy as jnp
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from streax.utils.typing import Array, Key


@struct.dataclass
class ObservationTracesWrapperState:
    mean: Array
    count: Array
    env_state: environment.EnvState


class ObservationTracesWrapper(GymnaxWrapper):
    def __init__(self, env, beta: float = 0.999):
        super().__init__(env)
        self.beta = beta

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, ObservationTracesWrapperState]:
        obs, env_state = self._env.reset(key, params)
        mean = (1.0 - self.beta) * obs
        count = jnp.float32(1.0)
        traced = mean / (1.0 - self.beta**count)
        state = ObservationTracesWrapperState(
            mean=mean,
            count=count,
            env_state=env_state,
        )
        return traced, state

    def step(
        self,
        key: Key,
        state: ObservationTracesWrapperState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, ObservationTracesWrapperState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        count = state.count + 1.0
        mean = self.beta * state.mean + (1.0 - self.beta) * obs
        traced = mean / (1.0 - self.beta**count)
        state = ObservationTracesWrapperState(
            mean=jnp.where(done, jnp.zeros_like(mean), mean),
            count=jnp.where(done, jnp.float32(0.0), count),
            env_state=env_state,
        )
        return traced, state, reward, done, info
