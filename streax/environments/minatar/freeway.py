"""Optimized Freeway-MinAtar, copied from gymnax and vectorized.

Dynamics, state layout and RNG stream match gymnax exactly; the per-car loops
in ``get_obs`` / ``step_cars`` / ``randomize_cars`` are vectorized. Only the
scalar agent-collision logic, which is genuinely sequential across cars, stays
as a tiny ``lax.scan``.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    pos: int
    cars: jax.Array
    move_timer: int
    time: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    player_speed: int = 3
    max_steps_in_episode: int = 2500


class Freeway(environment.Environment[EnvState, EnvParams]):
    """Optimized JAX implementation of Freeway MinAtar environment."""

    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (10, 10, 7)
        # Full action set: ['n','l','u','r','d','f']
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        # Minimal action set: ['n', 'u', 'd']
        self.minimal_action_set = jnp.array([0, 2, 4])
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
        """Perform single timestep state transition."""
        a = self.action_set[action]
        state, reward, win_cond = step_agent(a, state, params)

        key_speed, key_dirs = jax.random.split(key)
        speeds = jax.random.randint(key_speed, shape=(8,), minval=1, maxval=6)
        directions = jax.random.choice(key_dirs, jnp.array([-1, 1]), shape=(8,))
        win_cars = randomize_cars(speeds, directions, state.cars, False)
        state = state.replace(cars=jax.lax.select(win_cond, win_cars, state.cars))

        state = step_cars(state)

        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        state = state.replace(terminal=done)
        info = {"discount": self.discount(state, params)}
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
        """Reset environment state by sampling initial position."""
        key_speed, key_dirs = jax.random.split(key)
        speeds = jax.random.randint(key_speed, shape=(8,), minval=1, maxval=6)
        directions = jax.random.choice(key_dirs, jnp.array([-1, 1]), shape=(8,))
        state = EnvState(
            pos=9,
            cars=randomize_cars(speeds, directions, jnp.zeros((8, 4), dtype=int), True),
            move_timer=params.player_speed,
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        cars = state.cars
        x, y, direction = cars[:, 0], cars[:, 1], cars[:, 3]

        obs = jnp.zeros(self.obs_shape, dtype=jnp.float32)
        obs = obs.at[state.pos, 4, 0].set(1.0)
        obs = obs.at[y, x, 1].max(1.0)

        back_x = jnp.where(direction > 0, x - 1, x + 1)
        back_x = jnp.where(back_x < 0, 9, back_x)
        back_x = jnp.where(back_x > 9, 0, back_x)
        trail_channel = jnp.abs(direction) + 1
        obs = obs.at[y, back_x, trail_channel].max(1.0)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        return jnp.logical_and(
            params.max_steps_in_episode > 0,
            state.time > params.max_steps_in_episode,
        )

    @property
    def name(self) -> str:
        """Environment name."""
        return "Freeway-MinAtar"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return len(self.action_set)

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(len(self.action_set))

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(0, 1, self.obs_shape)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "pos": spaces.Discrete(10),
                "cars": spaces.Box(0, 1, jnp.zeros((8, 4)), dtype=jnp.int_),
                "move_timer": spaces.Discrete(params.player_speed),
                "time": spaces.Discrete(params.max_steps_in_episode),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(
    action: jax.Array, state: EnvState, params: EnvParams
) -> tuple[EnvState, jax.Array, jax.Array]:
    """Perform 1st part of step transition for agent."""
    cond_up = jnp.logical_and(action == 2, state.move_timer == 0)
    cond_down = jnp.logical_and(action == 4, state.move_timer == 0)
    any_cond = jnp.logical_or(cond_up, cond_down)
    state_up = jnp.maximum(0, state.pos - 1)
    state_down = jnp.minimum(9, state.pos + 1)
    pos = (1 - any_cond) * state.pos + cond_up * state_up + cond_down * state_down
    move_timer = jax.lax.select(any_cond, params.player_speed, state.move_timer)
    win_cond = pos == 0
    reward = win_cond * 1.0
    pos = jax.lax.select(win_cond, 9, pos)
    return state.replace(pos=pos, move_timer=move_timer), reward, win_cond


def step_cars(state: EnvState) -> EnvState:
    """Vectorized car update + scalar agent-collision scan."""
    cars = state.cars
    x, y, timer, direction = cars[:, 0], cars[:, 1], cars[:, 2], cars[:, 3]

    car_cond = timer == 0
    upd_timer = jnp.where(car_cond, jnp.abs(direction), timer)
    move = jnp.where(direction > 0, 1, -1)
    new_x = jnp.where(car_cond, x + move, x)
    new_x = jnp.where(jnp.logical_and(car_cond, new_x < 0), 9, new_x)
    new_x = jnp.where(jnp.logical_and(car_cond, new_x > 9), 0, new_x)
    final_timer = jnp.where(car_cond, upd_timer, timer - 1)

    new_cars = cars.at[:, 0].set(new_x).at[:, 2].set(final_timer)

    def body(pos, i):
        pos = jax.lax.select(jnp.logical_and(x[i] == 4, y[i] == pos), 9, pos)
        moved_hit = jnp.logical_and(
            car_cond[i], jnp.logical_and(new_x[i] == 4, y[i] == pos)
        )
        pos = jax.lax.select(moved_hit, 9, pos)
        return pos, None

    pos, _ = jax.lax.scan(body, state.pos, jnp.arange(8))
    move_timer = state.move_timer - (state.move_timer > 0)
    return state.replace(pos=pos, cars=new_cars, move_timer=move_timer)


def randomize_cars(
    speeds: jax.Array,
    directions: jax.Array,
    old_cars: jax.Array,
    initialize: bool,
) -> jax.Array:
    """Randomize car speeds & directions, vectorized over the 8 cars."""
    speeds_new = directions * speeds
    new_cars = jnp.stack(
        [
            jnp.zeros(8, dtype=jnp.int32),
            jnp.arange(1, 9, dtype=jnp.int32),
            jnp.abs(speeds_new).astype(jnp.int32),
            speeds_new.astype(jnp.int32),
        ],
        axis=1,
    )
    old_cars = old_cars.at[:, 2].set(jnp.abs(speeds_new)).at[:, 3].set(speeds_new)
    cars = jnp.where(initialize, new_cars, old_cars)
    return jnp.array(cars, dtype=jnp.int_)
