"""JAX implementation of the Breakout MinAtar environment.

Copied from gymnax's ``gymnax.environments.minatar.breakout``. The original is
already scatter-light, so it is kept as-is for a self-contained ``minatar``
namespace.

Source: github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    ball_y: jax.Array
    ball_x: jax.Array
    ball_dir: jax.Array
    pos: int
    brick_map: jax.Array
    strike: bool
    last_y: jax.Array
    last_x: jax.Array
    time: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    max_steps_in_episode: int = -1


class Breakout(environment.Environment[EnvState, EnvParams]):
    """JAX implementation of Breakout MinAtar environment."""

    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (10, 10, 4)
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        self.minimal_action_set = jnp.array([0, 1, 3])
        if use_minimal_action_set:
            self.action_set = self.minimal_action_set
        else:
            self.action_set = self.full_action_set

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        a = self.action_set[action]
        state, new_x, new_y = step_agent(state, a)
        state, reward = step_ball_brick(state, new_x, new_y)

        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        info = {}
        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        ball_start = jax.random.choice(key, jnp.array([0, 1]), shape=())
        state = EnvState(
            ball_y=jnp.array(3),
            ball_x=jnp.array([0, 9])[ball_start],
            ball_dir=jnp.array([2, 3])[ball_start],
            pos=4,
            brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
            strike=False,
            last_y=jnp.array(3),
            last_x=jnp.array([0, 9])[ball_start],
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        obs = jnp.zeros(self.obs_shape, dtype=bool)
        obs = obs.at[9, state.pos, 0].set(1)
        obs = obs.at[state.ball_y, state.ball_x, 1].set(1)
        obs = obs.at[state.last_y, state.last_x, 2].set(1)
        obs = obs.at[:, :, 3].set(state.brick_map)
        return obs.astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        capped = jnp.logical_and(
            params.max_steps_in_episode > 0,
            state.time >= params.max_steps_in_episode,
        )
        return jnp.logical_or(state.terminal, capped)

    @property
    def name(self) -> str:
        return "Breakout-MinAtar"

    @property
    def num_actions(self) -> int:
        return len(self.action_set)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "ball_y": spaces.Discrete(10),
                "ball_x": spaces.Discrete(10),
                "ball_dir": spaces.Discrete(10),
                "pos": spaces.Discrete(10),
                "brick_map": spaces.Box(0, 1, (10, 10)),
                "strike": spaces.Discrete(2),
                "last_y": spaces.Discrete(10),
                "last_x": spaces.Discrete(10),
                "time": spaces.Discrete(max(params.max_steps_in_episode, 0)),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(
    state: EnvState,
    action: jax.Array,
) -> tuple[EnvState, jax.Array, jax.Array]:
    pos = (
        jnp.maximum(0, state.pos - 1) * (action == 1)
        + jnp.minimum(9, state.pos + 1) * (action == 3)
        + state.pos * jnp.logical_and(action != 1, action != 3)
    )

    last_x = state.ball_x
    last_y = state.ball_y
    new_x = (
        (state.ball_x - 1) * (state.ball_dir == 0)
        + (state.ball_x + 1) * (state.ball_dir == 1)
        + (state.ball_x + 1) * (state.ball_dir == 2)
        + (state.ball_x - 1) * (state.ball_dir == 3)
    )
    new_y = (
        (state.ball_y - 1) * (state.ball_dir == 0)
        + (state.ball_y - 1) * (state.ball_dir == 1)
        + (state.ball_y + 1) * (state.ball_dir == 2)
        + (state.ball_y + 1) * (state.ball_dir == 3)
    )

    border_cond_x = jnp.logical_or(new_x < 0, new_x > 9)
    new_x = jax.lax.select(border_cond_x, (0 * (new_x < 0) + 9 * (new_x > 9)), new_x)
    ball_dir = jax.lax.select(
        border_cond_x, jnp.array([1, 0, 3, 2])[state.ball_dir], state.ball_dir
    )
    return (
        state.replace(
            pos=pos,
            last_x=last_x,
            last_y=last_y,
            ball_dir=ball_dir,
        ),
        new_x,
        new_y,
    )


def step_ball_brick(
    state: EnvState, new_x: jax.Array, new_y: jax.Array
) -> tuple[EnvState, jax.Array]:
    reward = 0

    border_cond1_y = new_y < 0
    new_y = jax.lax.select(border_cond1_y, 0, new_y)
    ball_dir = jax.lax.select(
        border_cond1_y, jnp.array([3, 2, 1, 0])[state.ball_dir], state.ball_dir
    )

    strike_toggle = jnp.logical_and(
        1 - border_cond1_y, state.brick_map[new_y, new_x] == 1
    )
    strike_bool = jnp.logical_and((1 - state.strike), strike_toggle)
    reward += strike_bool * 1.0

    brick_map = jax.lax.select(
        strike_bool, state.brick_map.at[new_y, new_x].set(0), state.brick_map
    )
    new_y = jax.lax.select(strike_bool, state.last_y, new_y)
    ball_dir = jax.lax.select(strike_bool, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir)

    brick_cond = jnp.logical_and(1 - strike_toggle, new_y == 9)

    spawn_bricks = jnp.logical_and(brick_cond, jnp.count_nonzero(brick_map) == 0)
    brick_map = jax.lax.select(spawn_bricks, brick_map.at[1:4, :].set(1), brick_map)

    redirect_ball1 = jnp.logical_and(brick_cond, state.ball_x == state.pos)
    ball_dir = jax.lax.select(
        redirect_ball1, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir
    )
    new_y = jax.lax.select(redirect_ball1, state.last_y, new_y)

    redirect_ball2a = jnp.logical_and(brick_cond, 1 - redirect_ball1)
    redirect_ball2 = jnp.logical_and(redirect_ball2a, new_x == state.pos)
    ball_dir = jax.lax.select(
        redirect_ball2, jnp.array([2, 3, 0, 1])[ball_dir], ball_dir
    )
    new_y = jax.lax.select(redirect_ball2, state.last_y, new_y)
    redirect_cond = jnp.logical_and(1 - redirect_ball1, 1 - redirect_ball2)
    terminal = jnp.logical_and(brick_cond, redirect_cond)

    strike = jax.lax.select(1 - strike_toggle == 1, False, True)
    return (
        state.replace(
            ball_dir=ball_dir,
            brick_map=brick_map,
            strike=strike,
            ball_x=new_x,
            ball_y=new_y,
            terminal=terminal,
        ),
        reward,
    )
