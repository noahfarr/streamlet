import itertools

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from gymnax.environments import spaces

from streamlet.utils.typing import Array, Key


@struct.dataclass
class Stimuli:
    active: Array
    onset: Array
    steps: Array


def stimuli(count: int) -> Stimuli:
    return Stimuli(
        active=jnp.zeros((count,), dtype=jnp.bool_),
        onset=jnp.full((count,), -1, dtype=jnp.int32),
        steps=jnp.zeros((count,), dtype=jnp.int32),
    )


def set_onset(state: Stimuli, mask: Array, time: Array) -> Stimuli:
    return state.replace(onset=jnp.where(mask, time, state.onset))


def tick(state: Stimuli, time: Array, length: int) -> Stimuli:
    expire = state.active & (state.steps >= length)
    starts = ~state.active & (state.onset == time)
    active = jnp.where(state.active, ~expire, starts)
    steps = jnp.where(
        state.active, jnp.where(expire, 0, state.steps) + 1, jnp.where(starts, 1, state.steps)
    )
    return Stimuli(active=active, onset=state.onset, steps=steps)


def values(state: Stimuli) -> Array:
    return state.active.astype(jnp.float32)


def randint(key: Key, low: int, high: int) -> Array:
    return jax.random.randint(key, (), low, high + 1)


@struct.dataclass
class ClassicalConditioningState:
    time: Array
    trial_start: Array
    us: Stimuli
    cs: Stimuli
    distractors: Stimuli


class TraceConditioning:
    def __init__(
        self,
        isi_interval: tuple[int, int],
        iti_interval: tuple[int, int],
        num_distractors: int = 0,
        activation_lengths: tuple[int, int, int] = (4, 2, 4),
    ):
        self.isi_interval = isi_interval
        self.iti_interval = iti_interval
        self.num_cs = 1
        self.num_distractors = num_distractors
        self.cs_length, self.us_length, self.distractor_length = activation_lengths
        self.distractor_probs = jnp.asarray(1.0 / np.arange(10, 10 * (num_distractors + 1), 10)[:num_distractors] if num_distractors else np.zeros(0), dtype=jnp.float32)

    @property
    def default_params(self) -> None:
        return None

    def configure_trial(self, key: Key, state: ClassicalConditioningState) -> ClassicalConditioningState:
        isi_key, iti_key = jax.random.split(key)
        isi = randint(isi_key, *self.isi_interval)
        iti = randint(iti_key, *self.iti_interval)
        cs = set_onset(state.cs, jnp.ones((1,), dtype=jnp.bool_), state.time)
        us = set_onset(state.us, jnp.ones((1,), dtype=jnp.bool_), state.time + isi)
        return state.replace(cs=cs, us=us, trial_start=state.time + isi + iti)

    def configure_distractors(self, key: Key, state: ClassicalConditioningState) -> ClassicalConditioningState:
        draws = jax.random.uniform(key, (self.num_distractors,))
        mask = ~state.distractors.active & (draws < self.distractor_probs)
        return state.replace(distractors=set_onset(state.distractors, mask, state.time))

    def tick(self, state: ClassicalConditioningState) -> ClassicalConditioningState:
        return state.replace(
            us=tick(state.us, state.time, self.us_length),
            cs=tick(state.cs, state.time, self.cs_length),
            distractors=tick(state.distractors, state.time, self.distractor_length),
        )

    def observation(self, state: ClassicalConditioningState) -> Array:
        return jnp.concatenate([values(state.us), values(state.cs), values(state.distractors)])

    def reset(self, key: Key, params=None) -> tuple[Array, ClassicalConditioningState]:
        trial_key, distractor_key = jax.random.split(key)
        state = ClassicalConditioningState(
            time=jnp.int32(0),
            trial_start=jnp.int32(0),
            us=stimuli(1),
            cs=stimuli(self.num_cs),
            distractors=stimuli(self.num_distractors),
        )
        state = self.configure_trial(trial_key, state)
        state = self.configure_distractors(distractor_key, state)
        state = self.tick(state)
        return self.observation(state), state

    def step(
        self, key: Key, state: ClassicalConditioningState, action: Array, params=None
    ) -> tuple[Array, ClassicalConditioningState, Array, Array, dict]:
        trial_key, distractor_key = jax.random.split(key)
        state = state.replace(time=state.time + 1)
        configured = self.configure_trial(trial_key, state)
        state = jax.tree.map(
            lambda new, old: jnp.where(state.time == state.trial_start, new, old), configured, state
        )
        state = self.configure_distractors(distractor_key, state)
        state = self.tick(state)
        cumulant = values(state.us)[0]
        return self.observation(state), state, cumulant, jnp.bool_(False), {}

    def observation_space(self, params=None) -> spaces.Box:
        return spaces.Box(low=0.0, high=1.0, shape=(1 + self.num_cs + self.num_distractors,))

    def action_space(self, params=None) -> spaces.Discrete:
        return spaces.Discrete(1)


def produce_activation_patterns(seed: int, num_cs: int, num_patterns: int) -> np.ndarray:
    generator = np.random.RandomState(seed)
    combinations = list(itertools.combinations(np.arange(num_cs), num_cs // 2))
    selected = generator.choice(len(combinations), size=num_patterns, replace=False)
    patterns = np.zeros((num_patterns, num_cs), dtype=np.float32)
    for row, index in enumerate(selected):
        patterns[row, list(combinations[index])] = 1.0
    return patterns


def binary_match(x: Array, patterns: Array) -> Array:
    ones = jnp.where(x.sum() == 0, 1.0, jnp.floor(patterns @ x / jnp.maximum(x.sum(), 1.0)))
    zeros = jnp.where(
        (1 - x).sum() == 0, 1.0, jnp.floor((1 - patterns) @ (1 - x) / jnp.maximum((1 - x).sum(), 1.0))
    )
    return ones * zeros


class TracePatterning(TraceConditioning):
    def __init__(
        self,
        isi_interval: tuple[int, int],
        iti_interval: tuple[int, int],
        num_cs: int = 2,
        num_activation_patterns: int = 1,
        activation_patterns_prob: float = 0.5,
        num_distractors: int = 0,
        activation_lengths: tuple[int, int, int] = (4, 2, 4),
        noise: float = 0.1,
        seed: int = 0,
    ):
        super().__init__(isi_interval, iti_interval, num_distractors, activation_lengths)
        self.num_cs = num_cs
        self.noise = noise
        self.activation_patterns = jnp.asarray(
            produce_activation_patterns(seed, num_cs, num_activation_patterns)
        )
        self.pattern_prob = (2**num_cs * activation_patterns_prob - num_activation_patterns) / (
            2**num_cs - num_activation_patterns
        )

    def configure_trial(self, key: Key, state: ClassicalConditioningState) -> ClassicalConditioningState:
        choice_key, index_key, random_key, isi_key, us_key, distractor_key, iti_key = jax.random.split(key, 7)
        stored = self.activation_patterns[
            jax.random.randint(index_key, (), 0, self.activation_patterns.shape[0])
        ]
        random_pattern = jax.random.randint(random_key, (self.num_cs,), 0, 2).astype(jnp.float32)
        pattern = jnp.where(jax.random.uniform(choice_key) < self.pattern_prob, stored, random_pattern)
        cs = set_onset(state.cs, pattern > 0, state.time)
        isi = randint(isi_key, *self.isi_interval)
        matches = binary_match(pattern, self.activation_patterns).sum() > 0
        draw = jax.random.uniform(us_key)
        deliver = jnp.where(matches, draw > self.noise, draw < self.noise)
        us = set_onset(state.us, jnp.full((1,), True) & deliver, state.time + isi)
        distractor_pattern = jax.random.randint(distractor_key, (self.num_distractors,), 0, 2) > 0
        distractors = set_onset(state.distractors, distractor_pattern, state.time)
        iti = randint(iti_key, *self.iti_interval)
        return state.replace(cs=cs, us=us, distractors=distractors, trial_start=state.time + isi + iti)

    def configure_distractors(self, key: Key, state: ClassicalConditioningState) -> ClassicalConditioningState:
        return state


class NoisyPatterning(TracePatterning):
    def __init__(self, isi_interval=None, iti_interval=(80, 120), activation_lengths=(4, 2, 4), **kwargs):
        cs_length = activation_lengths[0]
        super().__init__((cs_length, cs_length), iti_interval, activation_lengths=activation_lengths, **kwargs)


def isi_interval_for(isi: int) -> tuple[int, int]:
    third = isi // 3
    return (isi - third, isi + third)


def make(env_id: str, isi: int = 10, isi_interval=None, iti_interval=(80, 120), **kwargs) -> tuple:
    if isi_interval is None:
        isi_interval = isi_interval_for(isi)
    if env_id == "TraceConditioning":
        env = TraceConditioning(tuple(isi_interval), tuple(iti_interval), **kwargs)
    elif env_id == "TracePatterning":
        env = TracePatterning(tuple(isi_interval), tuple(iti_interval), **kwargs)
    elif env_id == "NoisyPatterning":
        env = NoisyPatterning(iti_interval=tuple(iti_interval), **kwargs)
    else:
        raise ValueError(f"Unknown classical conditioning benchmark {env_id}")
    return env, env.default_params
