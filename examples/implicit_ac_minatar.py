import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from stremax.algorithms import StreamAC, StreamACConfig
from stremax.environments import environment
from stremax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from stremax.loggers import DashboardLogger, MultiLogger
from stremax.networks import Flatten, heads, sparse
from stremax.optimizers import Implicit, ImplicitConfig, ObGD, ObGDConfig

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

actor_optimizer = ObGD(
    cfg=ObGDConfig(
        lr=1.0,
        kappa=3.0,
        beta2=0.999,
        eps=1e-8,
        adaptive=False,
    ),
)
critic_optimizer = Implicit(cfg=ImplicitConfig(lr=0.1))

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

logger = MultiLogger(
    [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "implicit-AC",
                "Environment": env_id,
                "Total Timesteps": f"{total_timesteps:_}",
            },
        ),
    ]
)

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

for i in range(num_epochs):
    start = time.perf_counter()
    key, train_key = jax.random.split(key)
    state, logs = train(jax.random.split(train_key, num_seeds), state, num_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_steps / (end - start))

    mask = logs.pop("returned_episode")
    axes = tuple(range(1, mask.ndim))
    episode_returns = jnp.mean(
        logs.pop("returned_episode_returns"), axis=axes, where=mask
    )
    episode_lengths = jnp.mean(
        logs.pop("returned_episode_lengths"), axis=axes, where=mask
    )
    discounted_episode_returns = jnp.mean(
        logs.pop("returned_discounted_episode_returns"), axis=axes, where=mask
    )

    data = {
        "training/SPS": SPS,
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        "training/discounted_episode_returns": discounted_episode_returns,
        **logs,
    }
    logger.log(data, step=state.step.mean(dtype=jnp.int32).item())

logger.finish()
