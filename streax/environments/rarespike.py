import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

from streax.utils.typing import Array, Key


@struct.dataclass
class RareSpikeParams:
    M: float = 3.0
    p: float = 0.03


@struct.dataclass
class RareSpikeState:
    step: int


def _sample_feature(key: Key, params: RareSpikeParams) -> Array:
    spike = jax.random.bernoulli(key, params.p)
    phi = jnp.where(spike, jnp.float32(params.M), jnp.float32(1.0))
    return jnp.reshape(phi, (1,))


class RareSpike:
    @property
    def default_params(self) -> RareSpikeParams:
        return RareSpikeParams()

    def reset(
        self, key: Key, params: RareSpikeParams | None = None
    ) -> tuple[Array, RareSpikeState]:
        params = params if params is not None else self.default_params
        obs = _sample_feature(key, params)
        return obs, RareSpikeState(step=0)

    def step(
        self,
        key: Key,
        state: RareSpikeState,
        action: Array,
        params: RareSpikeParams | None = None,
    ) -> tuple[Array, RareSpikeState, Array, Array, dict]:
        params = params if params is not None else self.default_params
        obs = _sample_feature(key, params)
        reward = jnp.float32(0.0)
        done = jnp.bool_(False)
        return obs, RareSpikeState(step=state.step + 1), reward, done, {}

    def observation_space(self, params: RareSpikeParams | None = None) -> spaces.Box:
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(1,))

    def action_space(self, params: RareSpikeParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(1)


def make(env_id: str = "Spike", M: float = 3.0, p: float = 0.03, **kwargs) -> tuple:
    env = RareSpike()
    return env, RareSpikeParams(M=M, p=p)
