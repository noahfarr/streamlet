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


class HostCallbackVectorEnv:
    def __init__(self, environment):
        self.env = environment
        observation_space = environment.single_observation_space
        action_space = environment.single_action_space
        self._action_dtype = canonicalize_dtype(action_space.dtype)
        self._observation = jax.ShapeDtypeStruct(
            observation_space.shape, canonicalize_dtype(observation_space.dtype)
        )
        self._reward = jax.ShapeDtypeStruct((), jnp.float32)
        self._done = jax.ShapeDtypeStruct((), jnp.bool_)

    def _host_reset(self, seed):
        seed = int(np.asarray(seed).reshape(-1)[0])
        observation, _ = self.env.reset() if seed < 0 else self.env.reset(seed=seed)
        return np.asarray(observation, dtype=self._observation.dtype)

    def _host_step(self, action):
        observation, reward, terminated, truncated, _ = self.env.step(
            np.asarray(action, dtype=self._action_dtype)
        )
        return (
            np.asarray(observation, dtype=self._observation.dtype),
            np.asarray(reward, dtype=np.float32),
            np.asarray(terminated, dtype=np.bool_),
            np.asarray(truncated, dtype=np.bool_),
        )

    def reset(self, seed):
        return jax.pure_callback(
            self._host_reset, self._observation, seed, vmap_method="broadcast_all"
        )

    def step(self, action):
        return jax.pure_callback(
            self._host_step,
            (self._observation, self._reward, self._done, self._done),
            jnp.asarray(action, dtype=self._action_dtype),
            vmap_method="broadcast_all",
        )


class GymnasiumWrapper:
    def __init__(
        self, environment, batch_shape: tuple[int, ...] = (1,), host_callback: bool = False
    ):
        if host_callback:
            self._environment = HostCallbackVectorEnv(environment)
        else:
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


def make(
    env_id, batch_shape: tuple[int, ...] = (1,), host_callback: bool = False, **kwargs
) -> tuple:
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
    return GymnasiumWrapper(environment, batch_shape=batch_shape, host_callback=host_callback), None
