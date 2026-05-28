import os
import urllib.request

import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from stremax.utils.typing import Array, Key

DATASET_URL = (
    "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/refs/heads/main/"
    "ETT-small/{env_id}.csv"
)


@struct.dataclass
class ETTState:
    current_step: int


class ETT:
    def __init__(self, env_id: str = "ETTm2", dataset_path: str | None = None):
        import numpy as np

        if dataset_path is None:
            dataset_path = f"{env_id}.csv"
        if not os.path.exists(dataset_path):
            urllib.request.urlretrieve(
                DATASET_URL.format(env_id=env_id), dataset_path
            )

        data = np.genfromtxt(
            dataset_path, delimiter=",", skip_header=1, usecols=range(1, 8)
        )
        ot = data[:, -1]
        self.add_value = float(ot.min())
        self.scaling_value = float(ot.max() - ot.min())
        normalized_ot = (ot - self.add_value) / self.scaling_value

        self.num_steps = data.shape[0]
        self.observations = jnp.asarray(data, dtype=jnp.float32)
        next_index = np.minimum(np.arange(self.num_steps) + 1, self.num_steps - 1)
        self.cumulants = jnp.asarray(normalized_ot[next_index], dtype=jnp.float32)

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, ETTState]:
        return self.observations[0], ETTState(current_step=0)

    def step(
        self, key: Key, state: ETTState, action: Array, params=None
    ) -> tuple[Array, ETTState, Array, Array, dict]:
        current_step = jnp.minimum(state.current_step + 1, self.num_steps - 1)
        obs = self.observations[current_step]
        reward = self.cumulants[current_step]
        done = current_step >= self.num_steps - 1
        return obs, ETTState(current_step=current_step), reward, done, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(7,))

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(1)


def make(env_id: str = "ETTm2", dataset_path: str | None = None, **kwargs) -> tuple:
    env = ETT(env_id=env_id, dataset_path=dataset_path)
    return env, env.default_params
