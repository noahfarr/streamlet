from typing import Any, Union

import jax.numpy as jnp
import lox
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from streamlet.utils.typing import Array, Key


@struct.dataclass
class NormalizeObservationWrapperState:
    mean: Array
    M2: Array
    count: float
    env_state: environment.EnvState

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env_state, name)


class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(
        self,
        env,
        eps: float = 1e-8,
        prior_count: float = 1.0,
        prior_var: float = 1.0,
    ):
        super().__init__(env)
        self.eps = eps
        self.prior_count = prior_count
        self.prior_var = prior_var

    def _welford_update(
        self, mean: Array, M2: Array, count: float, obs: Array
    ) -> tuple[Array, Array, float]:
        count = count + 1
        delta = obs - mean
        mean = mean + delta / count
        delta2 = obs - mean
        M2 = M2 + delta * delta2
        return mean, M2, count

    def _variance(self, M2: Array, count: float) -> Array:
        return M2 / count

    def _prior(self, obs: Array) -> tuple[Array, Array, float]:
        mean = jnp.zeros_like(obs)
        M2 = jnp.full_like(obs, self.prior_var) * self.prior_count
        return mean, M2, self.prior_count

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, NormalizeObservationWrapperState]:
        obs, env_state = self._env.reset(key, params)
        mean, M2, count = self._prior(obs)
        mean, M2, count = self._welford_update(mean, M2, count, obs)
        state = NormalizeObservationWrapperState(
            mean=mean,
            M2=M2,
            count=count,
            env_state=env_state,
        )
        var = self._variance(M2, count)
        return (obs - mean) / jnp.sqrt(var + self.eps), state

    def step(
        self,
        key: Key,
        state: NormalizeObservationWrapperState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, NormalizeObservationWrapperState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        mean, M2, count = self._welford_update(state.mean, state.M2, state.count, obs)
        state = NormalizeObservationWrapperState(
            mean=mean,
            M2=M2,
            count=count,
            env_state=env_state,
        )
        std = jnp.sqrt(self._variance(state.M2, state.count) + self.eps)
        lox.log(
            {
                "normalize_observation/mean": state.mean.mean(),
                "normalize_observation/std": std.mean(),
            }
        )
        return (
            (obs - state.mean) / std,
            state,
            reward,
            done,
            info,
        )
