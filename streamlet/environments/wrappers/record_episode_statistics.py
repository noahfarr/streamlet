from typing import Any

import lox
from flax import struct
from gymnax.environments import environment

from streamlet.utils.typing import Array, EnvParams, Key


@struct.dataclass
class RecordEpisodeStatisticsState:
    env_state: environment.EnvState
    episode_returns: float
    discounted_episode_returns: float
    episode_discount: float
    episode_lengths: int
    returned_episode_returns: float
    returned_discounted_episode_returns: float
    returned_episode_lengths: int
    reward_sum: float
    reward_rate: float
    total_steps: int

    def __getattr__(self, name):
        return getattr(self.env_state, name)

    @property
    def unwrapped(self):
        return getattr(self.env_state, "unwrapped", self.env_state)


class RecordEpisodeStatistics:
    def __init__(self, env, gamma: float = 0.99, reward_rate_beta: float = 1e-4):
        self._env = env
        self._gamma = gamma
        self._reward_rate_beta = reward_rate_beta

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(
        self, key: Key, params: EnvParams | None = None
    ) -> tuple[Array, RecordEpisodeStatisticsState]:
        obs, env_state = self._env.reset(key, params)
        state = RecordEpisodeStatisticsState(
            env_state, 0.0, 0.0, 1.0, 0, 0.0, 0.0, 0, 0.0, 0.0, 0
        )
        return obs, state

    def step(
        self,
        key: Key,
        state: RecordEpisodeStatisticsState,
        action: int | float,
        params: EnvParams | None = None,
    ) -> tuple[Array, RecordEpisodeStatisticsState, Array, bool, dict[str, Any]]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_discounted_episode_return = (
            state.discounted_episode_returns + state.episode_discount * reward
        )
        new_episode_discount = state.episode_discount * self._gamma
        new_episode_length = state.episode_lengths + 1
        new_total_steps = state.total_steps + 1
        new_reward_sum = state.reward_sum + reward
        beta = self._reward_rate_beta
        new_reward_rate = state.reward_rate + beta * (reward - state.reward_rate)
        state = RecordEpisodeStatisticsState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            discounted_episode_returns=new_discounted_episode_return * (1 - done),
            episode_discount=new_episode_discount * (1 - done) + done,
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_discounted_episode_returns=state.returned_discounted_episode_returns
            * (1 - done)
            + new_discounted_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            reward_sum=new_reward_sum,
            reward_rate=new_reward_rate,
            total_steps=new_total_steps,
        )
        lox.log(
            {
                "returned_episode_returns": state.returned_episode_returns,
                "returned_discounted_episode_returns": (
                    state.returned_discounted_episode_returns
                ),
                "returned_episode_lengths": state.returned_episode_lengths,
                "returned_episode": done,
                "average_reward": state.reward_sum / state.total_steps,
                "reward_rate": state.reward_rate
                / (1.0 - (1.0 - self._reward_rate_beta) ** state.total_steps),
            }
        )
        return obs, state, reward, done, info
