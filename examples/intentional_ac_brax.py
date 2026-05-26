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
)
from strelax.networks import heads, sparse
from strelax.optimizers import IntentionalOptimizer, IntentionalOptimizerConfig

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 1
env_id = "brax::halfcheetah"

gamma = 0.99
trace_lambda = 0.8

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

action_dim = env.action_space(env_params).shape[0]

config = StreamACConfig(
    num_envs=1,
    trace_lambda=trace_lambda,
    entropy_coefficient=0.01,
    gamma=gamma,
)

sparse_init = sparse(sparsity=0.9)
network = nn.Sequential(
    [
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
    ]
)

actor_network = nn.Sequential(
    [
        network,
        heads.StateDependentGaussian(
            action_dim=action_dim,
            transform=nn.softplus,
            kernel_init=sparse_init,
        ),
    ]
)

critic_network = nn.Sequential(
    [
        network,
        heads.VNetwork(kernel_init=sparse_init),
    ]
)

actor_optimizer = IntentionalOptimizer(
    cfg=IntentionalOptimizerConfig(
        gamma=gamma,
        trace_lambda=trace_lambda,
        eta=0.05,
    ),
)
critic_optimizer = IntentionalOptimizer(
    cfg=IntentionalOptimizerConfig(
        gamma=gamma,
        trace_lambda=trace_lambda,
        eta=0.5,
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
