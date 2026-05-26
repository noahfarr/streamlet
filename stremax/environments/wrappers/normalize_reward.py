from typing import Union

import jax.numpy as jnp
import lox
from flax import struct
from gymnax.environments import environment
from gymnax.wrappers.purerl import GymnaxWrapper

from stremax.utils.typing import Array, Key


@struct.dataclass
class NormalizeRewardWrapperState:
    mean: float
    M2: float
    count: float
    G: float
    env_state: environment.EnvState


class NormalizeRewardWrapper(GymnaxWrapper):
    def __init__(self, env, gamma: float = 0.99, eps: float = 1e-8):
        super().__init__(env)
        self.gamma = gamma
        self.eps = eps

    def _welford_update(
        self, mean: float, M2: float, count: float, value: Array
    ) -> tuple[Array, Array, float]:
        is_first = count == 0
        mean = jnp.where(is_first, value, mean)
        M2 = jnp.where(is_first, jnp.zeros_like(M2), M2)
        count = count + 1
        delta = value - mean
        mean = mean + delta / count
        delta2 = value - mean
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
    ) -> tuple[Array, NormalizeRewardWrapperState]:
        obs, env_state = self._env.reset(key, params)
        state = NormalizeRewardWrapperState(
            mean=0.0,
            M2=0.0,
            count=0.0,
            G=0.0,
            env_state=env_state,
        )
        return obs, state

    def step(
        self,
        key: Key,
        state: NormalizeRewardWrapperState,
        action: Union[int, float],
        params: environment.EnvParams | None = None,
    ) -> tuple[Array, NormalizeRewardWrapperState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        G = reward + self.gamma * state.G * (1 - done)

        mean, M2, count = self._welford_update(state.mean, state.M2, state.count, G)
        std = jnp.sqrt(self._variance(M2, count) + self.eps)
        scaled_reward = reward / std
        lox.log(
            {
                "normalize_reward/mean": mean,
                "normalize_reward/std": std,
            }
        )

        new_state = NormalizeRewardWrapperState(
            mean=mean,
            M2=M2,
            count=count,
            G=G * (1 - done),
            env_state=env_state,
        )
        return obs, new_state, scaled_reward, done, info
