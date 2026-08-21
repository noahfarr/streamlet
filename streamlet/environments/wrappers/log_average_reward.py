from typing import Any

import lox
from flax import struct
from gymnax.environments import environment

from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class LogAverageRewardState:
    env_state: environment.EnvState
    reward_sum: float
    reward_rate: float
    total_steps: int

    def __getattr__(self, name):
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class LogAverageReward:
    def __init__(self, env, beta: float = 1e-4):
        self._env = env
        self._beta = beta

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, LogAverageRewardState]:
        obs, env_state = self._env.reset(key, params)
        return obs, LogAverageRewardState(env_state, 0.0, 0.0, 0)

    def step(
        self,
        key: Key,
        state: LogAverageRewardState,
        action: int | float,
        params: EnvParams | None = None,
    ) -> tuple[Array, LogAverageRewardState, Array, bool, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        total_steps = state.total_steps + 1
        state = LogAverageRewardState(
            env_state=env_state,
            reward_sum=state.reward_sum + reward,
            reward_rate=state.reward_rate + self._beta * (reward - state.reward_rate),
            total_steps=total_steps,
        )
        lox.log(
            {
                "average_reward": state.reward_sum / total_steps,
                "reward_rate": state.reward_rate
                / (1.0 - (1.0 - self._beta) ** total_steps),
            }
        )
        return obs, state, reward, done, info
