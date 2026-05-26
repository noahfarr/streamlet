import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from strelax.algorithms import StreamQ, StreamQConfig
from strelax.environments import environment
from strelax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)
from strelax.networks import Flatten, heads, sparse
from strelax.optimizers import AdaptiveQ, AdaptiveQConfig

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "gymnax::Breakout-MinAtar"

gamma = 0.99
trace_lambda = 0.8

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

num_actions = env.action_space(env_params).n

config = StreamQConfig(
    num_envs=1,
    trace_lambda=trace_lambda,
    gamma=gamma,
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

q_network = nn.Sequential(
    [
        network,
        heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=sparse_init),
    ]
)

q_optimizer = AdaptiveQ(
    cfg=AdaptiveQConfig(
        gamma=gamma,
        trace_lambda=trace_lambda,
        eta=4.6e-4,
        eps=0.1,
        clip=1.0,
    ),
)

epsilon_start = 1.0
epsilon_end = 0.01
exploration_fraction = 0.2
exploration_steps = exploration_fraction * total_timesteps


def epsilon_schedule(step):
    frac = jnp.minimum(step / exploration_steps, 1.0)
    return epsilon_start + frac * (epsilon_end - epsilon_start)


agent = StreamQ(
    config,
    env,
    env_params,
    q_network,
    epsilon_schedule,
    q_optimizer,
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
