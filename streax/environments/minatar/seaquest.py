"""JAX port of the Seaquest MinAtar environment.

Ported from the official MinAtar numpy reference
(github.com/kenjyoung/MinAtar/blob/master/minatar/environments/seaquest.py).
Dynamic entity lists are represented as fixed-capacity arrays with an active
flag in the last column. Phases are applied in the same order as the official
``act`` so the rules match; rare same-cell tie-breaks may differ.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    oxygen: int
    diver_count: int
    sub_x: int
    sub_y: int
    sub_or: int
    f_bullets: jax.Array
    e_bullets: jax.Array
    e_fish: jax.Array
    e_subs: jax.Array
    divers: jax.Array
    e_spawn_speed: int
    e_spawn_timer: int
    d_spawn_timer: int
    move_speed: int
    ramp_index: int
    shot_timer: int
    surface: int
    terminal: bool


@struct.dataclass
class EnvParams(environment.EnvParams):
    ramping: bool = True
    ramp_interval: int = 100
    max_oxygen: int = 200
    init_spawn_speed: int = 20
    diver_spawn_speed: int = 30
    init_move_interval: int = 5
    shot_cool_down: int = 5
    enemy_shot_interval: int = 10
    enemy_move_interval: int = 5
    diver_move_interval: int = 5
    max_steps_in_episode: int = -1


def _occupancy(x: jax.Array, y: jax.Array, mask: jax.Array) -> jax.Array:
    xc = jnp.clip(x, 0, 9)
    yc = jnp.clip(y, 0, 9)
    return jnp.zeros((10, 10), dtype=bool).at[yc, xc].max(mask)


def _add(arr: jax.Array, row: jax.Array) -> jax.Array:
    free = arr[:, -1] == 0
    idx = jnp.argmax(free)
    return jax.lax.select(free.any(), arr.at[idx].set(row), arr)


class Seaquest(environment.Environment[EnvState, EnvParams]):
    """JAX port of Seaquest MinAtar environment."""

    def __init__(self, use_minimal_action_set: bool = True, capacity: int = 32):
        super().__init__()
        self.obs_shape = (10, 10, 10)
        self.capacity = capacity
        self.full_action_set = jnp.array([0, 1, 2, 3, 4, 5])
        self.minimal_action_set = jnp.array([0, 1, 2, 3, 4, 5])
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
        key_e, key_d = jax.random.split(key)

        state = maybe_spawn_enemy(key_e, state, params)
        state = maybe_spawn_diver(key_d, state, params)

        a = self.action_set[action]
        state = step_agent(state, a, params)

        state, reward = step_f_bullets(state)
        state = step_divers(state, params)
        state, reward = step_e_subs(state, reward, params)
        state = step_e_bullets(state)
        state, reward = step_e_fish(state, reward)
        state, reward = step_timers(state, reward, params)

        state = state.replace(time=state.time + 1)
        done = self.is_terminal(state, params)
        info = {"discount": self.discount(state, params)}
        return (
            jax.lax.stop_gradient(self.get_obs(state, params)),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        state = EnvState(
            oxygen=params.max_oxygen,
            diver_count=0,
            sub_x=5,
            sub_y=0,
            sub_or=0,
            f_bullets=jnp.zeros((self.capacity, 4), dtype=jnp.int32),
            e_bullets=jnp.zeros((self.capacity, 4), dtype=jnp.int32),
            e_fish=jnp.zeros((self.capacity, 5), dtype=jnp.int32),
            e_subs=jnp.zeros((self.capacity, 6), dtype=jnp.int32),
            divers=jnp.zeros((self.capacity, 5), dtype=jnp.int32),
            e_spawn_speed=params.init_spawn_speed,
            e_spawn_timer=params.init_spawn_speed,
            d_spawn_timer=params.diver_spawn_speed,
            move_speed=params.init_move_interval,
            ramp_index=0,
            shot_timer=0,
            surface=1,
            time=0,
            terminal=False,
        )
        return self.get_obs(state, params), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        if params is None:
            params = self.default_params
        obs = jnp.zeros(self.obs_shape, dtype=jnp.float32)
        obs = obs.at[state.sub_y, state.sub_x, 0].set(1.0)
        back_x = jnp.where(state.sub_or != 0, state.sub_x - 1, state.sub_x + 1)
        obs = obs.at[state.sub_y, back_x, 1].set(1.0)

        cells = jnp.arange(10)
        n_ox = jnp.maximum(state.oxygen, 0) * 10 // params.max_oxygen
        obs = obs.at[9, :, 7].set((cells < n_ox).astype(jnp.float32))
        diver_gauge = jnp.logical_and(cells >= 9 - state.diver_count, cells < 9)
        obs = obs.at[9, :, 8].set(diver_gauge.astype(jnp.float32))

        obs = _draw(obs, state.f_bullets, 2, trail=False)
        obs = _draw(obs, state.e_bullets, 4, trail=False)
        obs = _draw(obs, state.e_fish, 5, trail=True)
        obs = _draw(obs, state.e_subs, 6, trail=True)
        obs = _draw(obs, state.divers, 9, trail=True)
        return obs

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        capped = jnp.logical_and(
            params.max_steps_in_episode > 0,
            state.time >= params.max_steps_in_episode,
        )
        return jnp.logical_or(state.terminal, capped)

    @property
    def name(self) -> str:
        return "Seaquest-MinAtar"

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
                "oxygen": spaces.Discrete(params.max_oxygen),
                "diver_count": spaces.Discrete(20),
                "sub_x": spaces.Discrete(10),
                "sub_y": spaces.Discrete(10),
                "sub_or": spaces.Discrete(2),
                "f_bullets": spaces.Box(0, 1, (self.capacity, 4)),
                "e_bullets": spaces.Box(0, 1, (self.capacity, 4)),
                "e_fish": spaces.Box(0, 1, (self.capacity, 5)),
                "e_subs": spaces.Box(0, 1, (self.capacity, 6)),
                "divers": spaces.Box(0, 1, (self.capacity, 5)),
                "e_spawn_speed": spaces.Discrete(params.init_spawn_speed),
                "e_spawn_timer": spaces.Discrete(params.init_spawn_speed),
                "d_spawn_timer": spaces.Discrete(params.diver_spawn_speed),
                "move_speed": spaces.Discrete(1000),
                "ramp_index": spaces.Discrete(1000),
                "shot_timer": spaces.Discrete(params.shot_cool_down),
                "surface": spaces.Discrete(2),
                "time": spaces.Discrete(max(params.max_steps_in_episode, 0)),
                "terminal": spaces.Discrete(2),
            }
        )


def _draw(obs: jax.Array, ent: jax.Array, channel: int, trail: bool) -> jax.Array:
    x, y, active = ent[:, 0], ent[:, 1], ent[:, -1] != 0
    obs = obs.at[jnp.clip(y, 0, 9), jnp.clip(x, 0, 9), channel].max(
        active.astype(jnp.float32)
    )
    if trail:
        lr = ent[:, 2]
        back_x = jnp.where(lr != 0, x - 1, x + 1)
        in_frame = jnp.logical_and(back_x >= 0, back_x <= 9)
        bx = jnp.where(in_frame, back_x, 0)
        obs = obs.at[jnp.clip(y, 0, 9), bx, 3].max(
            jnp.logical_and(active, in_frame).astype(jnp.float32)
        )
    return obs


def maybe_spawn_enemy(key: jax.Array, state: EnvState, params: EnvParams) -> EnvState:
    key_lr, key_sub, key_y = jax.random.split(key, 3)
    lr = (jax.random.uniform(key_lr) < 0.5).astype(jnp.int32)
    is_sub = jax.random.uniform(key_sub) < (1 / 3)
    x = jnp.where(lr != 0, 0, 9)
    y = jax.random.randint(key_y, (), 1, 9)

    fish_block = jnp.logical_and(
        state.e_fish[:, -1] != 0,
        jnp.logical_and(state.e_fish[:, 1] == y, state.e_fish[:, 2] != lr),
    ).any()
    sub_block = jnp.logical_and(
        state.e_subs[:, -1] != 0,
        jnp.logical_and(state.e_subs[:, 1] == y, state.e_subs[:, 2] != lr),
    ).any()
    blocked = jnp.logical_or(fish_block, sub_block)

    do = jnp.logical_and(state.e_spawn_timer == 0, jnp.logical_not(blocked))
    add_sub = jnp.logical_and(do, is_sub)
    add_fish = jnp.logical_and(do, jnp.logical_not(is_sub))

    sub_row = jnp.array([x, y, lr, state.move_speed, params.enemy_shot_interval, 1])
    fish_row = jnp.array([x, y, lr, state.move_speed, 1])
    e_subs = jax.lax.select(add_sub, _add(state.e_subs, sub_row), state.e_subs)
    e_fish = jax.lax.select(add_fish, _add(state.e_fish, fish_row), state.e_fish)

    e_spawn_timer = jax.lax.select(
        state.e_spawn_timer == 0, state.e_spawn_speed, state.e_spawn_timer
    )
    return state.replace(e_subs=e_subs, e_fish=e_fish, e_spawn_timer=e_spawn_timer)


def maybe_spawn_diver(key: jax.Array, state: EnvState, params: EnvParams) -> EnvState:
    key_lr, key_y = jax.random.split(key)
    lr = (jax.random.uniform(key_lr) < 0.5).astype(jnp.int32)
    x = jnp.where(lr != 0, 0, 9)
    y = jax.random.randint(key_y, (), 1, 9)
    row = jnp.array([x, y, lr, params.diver_move_interval, 1])
    do = state.d_spawn_timer == 0
    divers = jax.lax.select(do, _add(state.divers, row), state.divers)
    d_spawn_timer = jax.lax.select(do, params.diver_spawn_speed, state.d_spawn_timer)
    return state.replace(divers=divers, d_spawn_timer=d_spawn_timer)


def step_agent(state: EnvState, action: jax.Array, params: EnvParams) -> EnvState:
    fire = jnp.logical_and(action == 5, state.shot_timer == 0)
    left = action == 1
    right = action == 3
    up = action == 2
    down = action == 4

    sub_x = jnp.where(left, jnp.maximum(0, state.sub_x - 1), state.sub_x)
    sub_x = jnp.where(right, jnp.minimum(9, sub_x + 1), sub_x)
    sub_or = jnp.where(left, 0, state.sub_or)
    sub_or = jnp.where(right, 1, sub_or)
    sub_y = jnp.where(up, jnp.maximum(0, state.sub_y - 1), state.sub_y)
    sub_y = jnp.where(down, jnp.minimum(8, sub_y + 1), sub_y)

    bullet = jnp.array([state.sub_x, state.sub_y, state.sub_or, 1])
    f_bullets = jax.lax.select(fire, _add(state.f_bullets, bullet), state.f_bullets)
    shot_timer = jax.lax.select(fire, params.shot_cool_down, state.shot_timer)
    return state.replace(
        sub_x=sub_x, sub_y=sub_y, sub_or=sub_or, f_bullets=f_bullets, shot_timer=shot_timer
    )


def step_f_bullets(state: EnvState) -> tuple[EnvState, jax.Array]:
    fb = state.f_bullets
    active = fb[:, 3] != 0
    new_x = jnp.where(active, jnp.where(fb[:, 2] != 0, fb[:, 0] + 1, fb[:, 0] - 1), fb[:, 0])
    oob = jnp.logical_and(active, jnp.logical_or(new_x < 0, new_x > 9))
    active = jnp.logical_and(active, jnp.logical_not(oob))
    fy = fb[:, 1]

    fish = state.e_fish
    subs = state.e_subs
    fish_active = fish[:, 4] != 0
    sub_active = subs[:, 5] != 0
    fish_map = _occupancy(fish[:, 0], fish[:, 1], fish_active)
    sub_map = _occupancy(subs[:, 0], subs[:, 1], sub_active)

    hit_fish = jnp.logical_and(active, fish_map[fy, jnp.clip(new_x, 0, 9)])
    hit_sub = jnp.logical_and(
        jnp.logical_and(active, jnp.logical_not(hit_fish)),
        sub_map[fy, jnp.clip(new_x, 0, 9)],
    )
    bullet_fish_map = _occupancy(new_x, fy, hit_fish)
    bullet_sub_map = _occupancy(new_x, fy, hit_sub)

    fish_die = jnp.logical_and(fish_active, bullet_fish_map[fish[:, 1], fish[:, 0]])
    sub_die = jnp.logical_and(sub_active, bullet_sub_map[subs[:, 1], subs[:, 0]])
    reward = jnp.sum(fish_die) + jnp.sum(sub_die)

    consumed = jnp.logical_or(hit_fish, hit_sub)
    active = jnp.logical_and(active, jnp.logical_not(consumed))
    fb = fb.at[:, 0].set(new_x).at[:, 3].set(active.astype(jnp.int32))
    fish = fish.at[:, 4].set(
        jnp.logical_and(fish_active, jnp.logical_not(fish_die)).astype(jnp.int32)
    )
    subs = subs.at[:, 5].set(
        jnp.logical_and(sub_active, jnp.logical_not(sub_die)).astype(jnp.int32)
    )
    return state.replace(f_bullets=fb, e_fish=fish, e_subs=subs), reward


def _pickup(active, at_sub, diver_count):
    elig = jnp.logical_and(active, at_sub)
    csum = jnp.cumsum(elig.astype(jnp.int32)) - elig.astype(jnp.int32)
    room = 6 - diver_count
    picked = jnp.logical_and(elig, csum < room)
    return picked, diver_count + jnp.sum(picked)


def step_divers(state: EnvState, params: EnvParams) -> EnvState:
    d = state.divers
    active = d[:, 4] != 0
    x, y, lr, mt = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    at_sub = jnp.logical_and(x == state.sub_x, y == state.sub_y)

    picked_pre, dc = _pickup(active, at_sub, state.diver_count)
    active = jnp.logical_and(active, jnp.logical_not(picked_pre))

    move_cond = jnp.logical_and(active, mt == 0)
    new_mt = jnp.where(active, jnp.where(mt == 0, params.diver_move_interval, mt - 1), mt)
    new_x = jnp.where(move_cond, jnp.where(lr != 0, x + 1, x - 1), x)
    oob = jnp.logical_and(move_cond, jnp.logical_or(new_x < 0, new_x > 9))
    active = jnp.logical_and(active, jnp.logical_not(oob))

    at_sub_post = jnp.logical_and(
        move_cond, jnp.logical_and(new_x == state.sub_x, y == state.sub_y)
    )
    picked_post, dc = _pickup(active, at_sub_post, dc)
    active = jnp.logical_and(active, jnp.logical_not(picked_post))

    d = d.at[:, 0].set(new_x).at[:, 3].set(new_mt).at[:, 4].set(active.astype(jnp.int32))
    return state.replace(divers=d, diver_count=dc)


def step_e_subs(state: EnvState, reward: jax.Array, params: EnvParams) -> tuple[EnvState, jax.Array]:
    s = state.e_subs
    active = s[:, 5] != 0
    x, y, lr, mt, st = s[:, 0], s[:, 1], s[:, 2], s[:, 3], s[:, 4]
    at_player = jnp.logical_and(x == state.sub_x, y == state.sub_y)
    terminal = jnp.logical_or(state.terminal, jnp.logical_and(active, at_player).any())

    move_cond = jnp.logical_and(active, mt == 0)
    new_mt = jnp.where(active, jnp.where(mt == 0, state.move_speed, mt - 1), mt)
    new_x = jnp.where(move_cond, jnp.where(lr != 0, x + 1, x - 1), x)
    oob = jnp.logical_and(move_cond, jnp.logical_or(new_x < 0, new_x > 9))
    in_frame = jnp.logical_and(move_cond, jnp.logical_not(oob))

    moved_hit_player = jnp.logical_and(
        in_frame, jnp.logical_and(new_x == state.sub_x, y == state.sub_y)
    )
    terminal = jnp.logical_or(terminal, moved_hit_player.any())

    fb = state.f_bullets
    fb_active = fb[:, 3] != 0
    fb_map = _occupancy(fb[:, 0], fb[:, 1], fb_active)
    sub_collide = jnp.logical_and(
        jnp.logical_and(in_frame, jnp.logical_not(moved_hit_player)),
        fb_map[y, jnp.clip(new_x, 0, 9)],
    )
    sub_kill_map = _occupancy(new_x, y, sub_collide)
    fb_die = jnp.logical_and(fb_active, sub_kill_map[fb[:, 1], fb[:, 0]])
    reward = reward + jnp.sum(sub_collide)

    removed = jnp.logical_or(oob, sub_collide)
    new_active = jnp.logical_and(active, jnp.logical_not(removed))

    shoot = jnp.logical_and(jnp.logical_and(active, st == 0), jnp.logical_not(oob))
    new_st = jnp.where(active, jnp.where(st == 0, params.enemy_shot_interval, st - 1), st)

    e_bullets = state.e_bullets
    idx = jnp.arange(s.shape[0])
    shoot_csum = jnp.cumsum(shoot.astype(jnp.int32)) - shoot.astype(jnp.int32)
    free = e_bullets[:, 3] == 0
    free_csum = jnp.cumsum(free.astype(jnp.int32)) - free.astype(jnp.int32)

    def add_bullet(eb, i):
        row = jnp.array([new_x[i], y[i], lr[i], 1])
        target = jnp.argmax(jnp.logical_and(free, free_csum == shoot_csum[i]))
        return jax.lax.select(
            jnp.logical_and(shoot[i], jnp.logical_and(free, free_csum == shoot_csum[i]).any()),
            eb.at[target].set(row),
            eb,
        ), None

    e_bullets, _ = jax.lax.scan(add_bullet, e_bullets, idx)

    s = s.at[:, 0].set(new_x).at[:, 3].set(new_mt).at[:, 4].set(new_st).at[:, 5].set(
        new_active.astype(jnp.int32)
    )
    fb = fb.at[:, 3].set(
        jnp.logical_and(fb_active, jnp.logical_not(fb_die)).astype(jnp.int32)
    )
    return state.replace(e_subs=s, f_bullets=fb, e_bullets=e_bullets, terminal=terminal), reward


def step_e_bullets(state: EnvState) -> EnvState:
    b = state.e_bullets
    active = b[:, 3] != 0
    x, y, d = b[:, 0], b[:, 1], b[:, 2]
    at_player = jnp.logical_and(x == state.sub_x, y == state.sub_y)
    terminal = jnp.logical_or(state.terminal, jnp.logical_and(active, at_player).any())

    new_x = jnp.where(active, jnp.where(d != 0, x + 1, x - 1), x)
    oob = jnp.logical_and(active, jnp.logical_or(new_x < 0, new_x > 9))
    active = jnp.logical_and(active, jnp.logical_not(oob))
    moved_hit = jnp.logical_and(
        active, jnp.logical_and(new_x == state.sub_x, y == state.sub_y)
    )
    terminal = jnp.logical_or(terminal, moved_hit.any())

    b = b.at[:, 0].set(new_x).at[:, 3].set(active.astype(jnp.int32))
    return state.replace(e_bullets=b, terminal=terminal)


def step_e_fish(state: EnvState, reward: jax.Array) -> tuple[EnvState, jax.Array]:
    f = state.e_fish
    active = f[:, 4] != 0
    x, y, lr, mt = f[:, 0], f[:, 1], f[:, 2], f[:, 3]
    at_player = jnp.logical_and(x == state.sub_x, y == state.sub_y)
    terminal = jnp.logical_or(state.terminal, jnp.logical_and(active, at_player).any())

    move_cond = jnp.logical_and(active, mt == 0)
    new_mt = jnp.where(active, jnp.where(mt == 0, state.move_speed, mt - 1), mt)
    new_x = jnp.where(move_cond, jnp.where(lr != 0, x + 1, x - 1), x)
    oob = jnp.logical_and(move_cond, jnp.logical_or(new_x < 0, new_x > 9))
    in_frame = jnp.logical_and(move_cond, jnp.logical_not(oob))

    moved_hit_player = jnp.logical_and(
        in_frame, jnp.logical_and(new_x == state.sub_x, y == state.sub_y)
    )
    terminal = jnp.logical_or(terminal, moved_hit_player.any())

    fb = state.f_bullets
    fb_active = fb[:, 3] != 0
    fb_map = _occupancy(fb[:, 0], fb[:, 1], fb_active)
    fish_collide = jnp.logical_and(
        jnp.logical_and(in_frame, jnp.logical_not(moved_hit_player)),
        fb_map[y, jnp.clip(new_x, 0, 9)],
    )
    fish_kill_map = _occupancy(new_x, y, fish_collide)
    fb_die = jnp.logical_and(fb_active, fish_kill_map[fb[:, 1], fb[:, 0]])
    reward = reward + jnp.sum(fish_collide)

    removed = jnp.logical_or(oob, fish_collide)
    new_active = jnp.logical_and(active, jnp.logical_not(removed))
    f = f.at[:, 0].set(new_x).at[:, 3].set(new_mt).at[:, 4].set(
        new_active.astype(jnp.int32)
    )
    fb = fb.at[:, 3].set(
        jnp.logical_and(fb_active, jnp.logical_not(fb_die)).astype(jnp.int32)
    )
    return state.replace(e_fish=f, f_bullets=fb, terminal=terminal), reward


def step_timers(state: EnvState, reward: jax.Array, params: EnvParams) -> tuple[EnvState, jax.Array]:
    e_spawn_timer = state.e_spawn_timer - (state.e_spawn_timer > 0)
    d_spawn_timer = state.d_spawn_timer - (state.d_spawn_timer > 0)
    shot_timer = state.shot_timer - (state.shot_timer > 0)
    terminal = jnp.logical_or(state.terminal, state.oxygen <= 0)

    above = state.sub_y > 0
    below = jnp.logical_not(above)
    do_surface_logic = jnp.logical_and(below, state.surface == 0)
    no_diver = jnp.logical_and(do_surface_logic, state.diver_count == 0)
    do_surface = jnp.logical_and(do_surface_logic, state.diver_count != 0)
    terminal = jnp.logical_or(terminal, no_diver)

    full = state.diver_count == 6
    surface_r = jnp.where(full, state.oxygen * 10 // params.max_oxygen, 0)
    diver_after = jnp.where(full, 0, state.diver_count) - 1

    ramp_cond = jnp.logical_and(
        params.ramping,
        jnp.logical_or(state.e_spawn_speed > 1, state.move_speed > 2),
    )
    move_dec = jnp.logical_and(
        ramp_cond, jnp.logical_and(state.move_speed > 2, state.ramp_index % 2)
    )
    espawn_dec = jnp.logical_and(ramp_cond, state.e_spawn_speed > 1)

    oxygen = jnp.where(above, state.oxygen - 1, jnp.where(do_surface, params.max_oxygen, state.oxygen))
    surface = jnp.where(above, 0, 1)
    diver_count = jnp.where(do_surface, diver_after, state.diver_count)
    move_speed = jnp.where(do_surface, state.move_speed - move_dec, state.move_speed)
    e_spawn_speed = jnp.where(do_surface, state.e_spawn_speed - espawn_dec, state.e_spawn_speed)
    ramp_index = jnp.where(do_surface, state.ramp_index + ramp_cond, state.ramp_index)
    reward = reward + jnp.where(do_surface, surface_r, 0)

    return (
        state.replace(
            oxygen=oxygen,
            surface=surface,
            diver_count=diver_count,
            move_speed=move_speed,
            e_spawn_speed=e_spawn_speed,
            ramp_index=ramp_index,
            e_spawn_timer=e_spawn_timer,
            d_spawn_timer=d_spawn_timer,
            shot_timer=shot_timer,
            terminal=terminal,
        ),
        reward,
    )
