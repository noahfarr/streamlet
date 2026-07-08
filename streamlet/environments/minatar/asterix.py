from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    player_x: int
    player_y: int
    shot_timer: int
    spawn_speed: int
    spawn_timer: int
    move_speed: int
    move_timer: int
    ramp_timer: int
    ramp_index: int
    entities: jax.Array
    time: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    ramping: bool = True
    ramp_interval: int = 100
    init_spawn_speed: int = 10
    init_move_interval: int = 5
    shot_cool_down: int = 5
    max_steps_in_episode: int = -1


class Asterix(environment.Environment[EnvState, EnvParams]):
    def __init__(self, use_minimal_action_set: bool = True):
        super().__init__()
        self.obs_shape = (10, 10, 4)
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        self.minimal_action_set = jnp.array([0, 1, 2, 3, 4])
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
        spawn_entities_now = state.spawn_timer == 0
        entity, slot = spawn_entity(key, state)
        should_spawn = jnp.logical_and(spawn_entities_now, entity[4] != 0)
        entities = jax.lax.select(
            should_spawn,
            state.entities.at[slot].set(entity),
            state.entities,
        )
        spawn_timer = jax.lax.select(
            spawn_entities_now, state.spawn_speed, state.spawn_timer
        )
        state = state.replace(entities=entities, spawn_timer=spawn_timer)

        a = self.action_set[action]
        state = step_agent(state, a)

        state, reward, done = step_entities(state)

        state = step_timers(state, params)

        state = state.replace(time=state.time + 1, terminal=done)
        done = self.is_terminal(state, params)
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
        state = EnvState(
            player_x=5,
            player_y=5,
            shot_timer=0,
            spawn_speed=params.init_spawn_speed,
            spawn_timer=params.init_spawn_speed,
            move_speed=params.init_move_interval,
            move_timer=params.init_move_interval,
            ramp_timer=params.ramp_interval,
            ramp_index=0,
            entities=jnp.zeros((8, 5), dtype=int),
            time=0,
            terminal=False,
        )
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        e = state.entities
        x, y = e[:, 0], e[:, 1]
        lr, gold, filled = e[:, 2], e[:, 3], e[:, 4] != 0

        obs = jnp.zeros((10, 10, 4), dtype=jnp.float32)
        obs = obs.at[state.player_y, state.player_x, 0].set(1.0)

        enemy = jnp.logical_and(filled, gold == 0).astype(jnp.float32)
        treasure = jnp.logical_and(filled, gold != 0).astype(jnp.float32)
        obs = obs.at[y, x, 1].max(enemy)
        obs = obs.at[y, x, 3].max(treasure)

        back_x = jnp.where(lr != 0, x - 1, x + 1)
        in_frame = jnp.logical_and(back_x >= 0, back_x <= 9)
        back_x = jnp.where(in_frame, back_x, 0)
        trail = jnp.logical_and(filled, in_frame).astype(jnp.float32)
        obs = obs.at[y, back_x, 2].max(trail)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        capped = jnp.logical_and(
            params.max_steps_in_episode > 0,
            state.time >= params.max_steps_in_episode,
        )
        return jnp.logical_or(state.terminal, capped)

    @property
    def name(self) -> str:
        return "Asterix-MinAtar"

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
                "player_x": spaces.Discrete(10),
                "player_y": spaces.Discrete(10),
                "shot_timer": spaces.Discrete(1000),
                "spawn_speed": spaces.Discrete(1000),
                "spawn_timer": spaces.Discrete(1000),
                "move_speed": spaces.Discrete(1000),
                "move_timer": spaces.Discrete(1000),
                "ramp_timer": spaces.Discrete(1000),
                "ramp_index": spaces.Discrete(1000),
                "entities": spaces.Box(0, 1, (8, 5)),
                "time": spaces.Discrete(max(params.max_steps_in_episode, 0)),
                "terminal": spaces.Discrete(2),
            }
        )


def step_agent(state: EnvState, action: jax.Array) -> EnvState:
    player_x = (
        jnp.maximum(0, state.player_x - 1) * (action == 1)
        + jnp.minimum(9, state.player_x + 1) * (action == 3)
        + state.player_x * jnp.logical_and(action != 1, action != 3)
    )

    player_y = (
        jnp.maximum(1, state.player_y - 1) * (action == 2)
        + jnp.minimum(8, state.player_y + 1) * (action == 4)
        + state.player_y * jnp.logical_and(action != 2, action != 4)
    )
    return state.replace(player_x=player_x, player_y=player_y)


def spawn_entity(key: jax.Array, state: EnvState) -> tuple[jax.Array, jax.Array]:
    key_lr, key_gold, key_slot = jax.random.split(key, 3)
    lr = jax.random.choice(key_lr, jnp.array([1, 0]))
    is_gold = jax.random.choice(
        key_gold, jnp.array([1, 0]), p=jnp.array([1 / 3, 2 / 3])
    )
    x = (1 - lr) * 9
    slot, free = sample_free_slot(key_slot, state.entities[:, 4])
    entity = jnp.array([x, slot + 1, lr, is_gold, free])
    return entity, slot


def sample_free_slot(
    key: jax.Array, state_entities: jax.Array
) -> tuple[jax.Array, jax.Array]:
    free = state_entities == 0
    r = jax.random.uniform(key, (8,))
    slot_id = jnp.argmax(jnp.where(free, r, -1.0))
    free_slot = free.any().astype(jnp.int32)
    return slot_id, free_slot


def step_entities(state: EnvState) -> tuple[EnvState, jax.Array, jax.Array]:
    entities = state.entities
    px, py = state.player_x, state.player_y

    x, y = entities[:, 0], entities[:, 1]
    gold, filled = entities[:, 3], entities[:, 4] != 0
    collision = jnp.logical_and(jnp.logical_and(x == px, y == py), filled)
    collision_gold = jnp.logical_and(collision, gold != 0)
    reward = jnp.sum(collision_gold)
    done = jnp.sum(jnp.logical_and(collision, gold == 0))
    entities = entities * (1 - collision_gold.astype(entities.dtype))[:, None]

    time_to_move = state.move_timer == 0
    move_timer = jax.lax.select(time_to_move, state.move_speed, state.move_timer)

    x0 = entities[:, 0]
    lr = entities[:, 2]
    gold = entities[:, 3]
    filled = entities[:, 4] != 0
    step = 2 * lr - 1
    moved_x = jnp.where(filled, x0 + step, x0)
    outside = jnp.logical_or(moved_x < 0, moved_x > 9)
    keep = jnp.logical_and(filled, jnp.logical_not(outside))

    moved = entities.at[:, 0].set(moved_x) * keep.astype(entities.dtype)[:, None]

    collision = jnp.logical_and(
        jnp.logical_and(moved_x == px, entities[:, 1] == py), filled
    )
    collision_gold = jnp.logical_and(collision, gold != 0)
    moved = moved * (1 - collision_gold.astype(entities.dtype))[:, None]
    reward_move = jnp.sum(collision_gold)
    done_move = jnp.sum(jnp.logical_and(collision, gold == 0))

    entities = jax.lax.select(time_to_move, moved, entities)
    reward = reward + jax.lax.select(time_to_move, reward_move, jnp.array(0))
    done = done + jax.lax.select(time_to_move, done_move, jnp.array(0))

    return (
        state.replace(entities=entities, move_timer=move_timer),
        reward,
        jnp.bool_(done > 0),
    )


def step_timers(state: EnvState, params: EnvParams) -> EnvState:
    spawn_timer = state.spawn_timer - 1
    move_timer = state.move_timer - 1

    ramp_cond = jnp.logical_and(
        params.ramping,
        jnp.logical_or(state.spawn_speed > 1, state.move_speed > 1),
    )
    do_ramp = jnp.logical_and(ramp_cond, state.ramp_timer < 0)
    ramp_timer = jax.lax.select(
        ramp_cond,
        jax.lax.select(
            state.ramp_timer >= 0, state.ramp_timer - 1, params.ramp_interval
        ),
        state.ramp_timer,
    )
    move_speed_cond = jnp.logical_and(
        do_ramp, jnp.logical_and(state.move_speed > 1, state.ramp_index % 2)
    )
    move_speed = state.move_speed - move_speed_cond
    spawn_speed_cond = jnp.logical_and(do_ramp, state.spawn_speed > 1)
    spawn_speed = state.spawn_speed - spawn_speed_cond
    ramp_index = state.ramp_index + do_ramp
    return state.replace(
        spawn_timer=spawn_timer,
        move_timer=move_timer,
        ramp_timer=ramp_timer,
        move_speed=move_speed,
        spawn_speed=spawn_speed,
        ramp_index=ramp_index,
    )
