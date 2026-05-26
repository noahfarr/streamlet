from typing import Union

import jax.numpy as jnp
import lox
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from stremax.utils.typing import Array, Key


@struct.dataclass
class NormalizeObservationWrapperState:
    mean: Array
    M2: Array
    count: float
    env_state: environment.EnvState


class NormalizeObservationWrapper(GymnaxWrapper):
    def __init__(self, env, eps: float = 1e-8):
        super().__init__(env)
        self.eps = eps

    def _welford_update(
        self, mean: Array, M2: Array, count: float, obs: Array
    ) -> tuple[Array, Array, float]:
        is_first = count == 0
        mean = jnp.where(is_first, obs, mean)
        M2 = jnp.where(is_first, jnp.zeros_like(M2), M2)
        count = count + 1
        delta = obs - mean
        mean = mean + delta / count
        delta2 = obs - mean
        M2 = M2 + delta * delta2
        return mean, M2, count

    def _variance(self, M2: Array, count: float) -> Array:
        # Sample (Bessel-corrected) variance, with var=1 until two samples seen,
        # matching the reference SampleMeanStd estimator.
        return jnp.where(
            count < 2, jnp.ones_like(M2), M2 / jnp.maximum(count - 1.0, 1.0)
        )

    def reset(
        self, key: Key, params: environment.EnvParams | None = None
    ) -> tuple[Array, NormalizeObservationWrapperState]:
        obs, env_state = self._env.reset(key, params)
        mean = jnp.zeros_like(obs)
        M2 = jnp.zeros_like(obs)
        count = 0.0
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
