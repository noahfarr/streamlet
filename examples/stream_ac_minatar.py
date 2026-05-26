import time
from dataclasses import asdict

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from strelax.algorithms import StreamAC, StreamACConfig
from strelax.environments import environment
from strelax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from strelax.networks import Flatten, heads, sparse
from strelax.optimizers import OBGD, OBGDConfig

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "gymnax::Breakout-MinAtar"

env, env_params = environment.make(env_id)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env)

num_actions = env.action_space(env_params).n

config = StreamACConfig(
    num_envs=1,
    trace_lambda=0.8,
    entropy_coefficient=0.01,
    gamma=0.99,
)

sparse_init = sparse(sparsity=0.9)
network = nn.Sequential(
    [
        nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        Flatten(start_dim=-3),
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
    ]
)

actor_network = nn.Sequential(
    [
        network,
        heads.Categorical(action_dim=num_actions, kernel_init=sparse_init),
    ]
)

critic_network = nn.Sequential(
    [
        network,
        heads.VNetwork(kernel_init=sparse_init),
    ]
)

actor_optimizer = OBGD(
    cfg=OBGDConfig(
        lr=1.0,
        kappa=3.0,
        beta2=0.999,
        eps=1e-8,
        adaptive=False,
    ),
)
critic_optimizer = OBGD(
    cfg=OBGDConfig(
        lr=1.0,
        kappa=2.0,
        beta2=0.999,
        eps=1e-8,
        adaptive=False,
    ),
)

agent = StreamAC(
    config,
    env,
    env_params,
    actor_network,
    critic_network,
    actor_optimizer,
    critic_optimizer,
)


init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

train_keys = jax.random.split(key, num_epochs)
for i in range(num_epochs):
    state, logs = train(jax.random.split(train_keys[i], num_seeds), state, num_steps)

    returned_episode = logs.pop("returned_episode")
    episode_statistics = {
        "episode_returns": logs.pop("returned_episode_returns"),
        "episode_lengths": logs.pop("returned_episode_lengths"),
        "discounted_episode_returns": logs.pop("returned_discounted_episode_returns"),
    }

    data = {}
    if returned_episode.any():
        data |= {
            name: jnp.mean(value, where=returned_episode, axis=(1, 2))
            for name, value in episode_statistics.items()
        }
    print(f"epoch {i + 1}/{num_epochs}: {data}")
