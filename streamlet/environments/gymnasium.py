import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from streamlet.utils import canonicalize_dtype
from streamlet.utils.typing import Array, Key


@struct.dataclass
class GymnasiumState:
    step: int = 0


class GymnasiumWrapper:
    def __init__(self, environment, batch_shape: tuple[int, ...] = (1,)):
        import gymnasium_ffi

        self._environment = gymnasium_ffi.VectorEnv(environment)
        self.batch_shape = tuple(batch_shape)

        observation_space = environment.single_observation_space
        self.observation_shape = observation_space.shape
        self.observation_dtype = canonicalize_dtype(observation_space.dtype)

        action_space = environment.single_action_space
        self.discrete = hasattr(action_space, "n")
        if self.discrete:
            self.num_actions = int(action_space.n)
        else:
            self.action_shape = action_space.shape
            self.action_dtype = canonicalize_dtype(action_space.dtype)
            self._action_low = np.asarray(action_space.low, dtype=np.float32)
            self._action_high = np.asarray(action_space.high, dtype=np.float32)

    @property
    def default_params(self) -> None:
        return None

    def reset(self, key: Key, params=None) -> tuple[Array, GymnasiumState]:
        bits = jax.random.key_data(key).sum().astype(jnp.int32)
        seed = 0 * bits - 1

        observation = self._environment.reset(seed)
        state = GymnasiumState(step=0)
        return observation, state

    def step(
        self,
        key: Key,
        state: GymnasiumState,
        action: Array,
        params=None,
    ) -> tuple[Array, GymnasiumState, Array, Array, dict]:
        observation, rewards, terminations, truncations = self._environment.step(
            action
        )
        dones = terminations | truncations

        new_state = GymnasiumState(step=state.step + 1)
        return observation, new_state, rewards, dones, {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=self.observation_shape,
            dtype=self.observation_dtype,
        )

    def action_space(self, params=None):
        if self.discrete:
            return spaces.Discrete(self.num_actions)
        return spaces.Box(
            low=jnp.asarray(self._action_low),
            high=jnp.asarray(self._action_high),
            shape=self.action_shape,
            dtype=self.action_dtype,
        )


def make(env_id, batch_shape: tuple[int, ...] = (1,), **kwargs) -> tuple:
    import gymnasium
    from gymnasium.vector import AutoresetMode

    num_envs = int(np.prod(batch_shape))
    vector_kwargs = {
        "autoreset_mode": AutoresetMode.SAME_STEP,
        **kwargs.pop("vector_kwargs", {}),
    }
    environment = gymnasium.make_vec(
        env_id, num_envs=num_envs, vector_kwargs=vector_kwargs, **kwargs
    )
    return GymnasiumWrapper(environment, batch_shape=batch_shape), None
