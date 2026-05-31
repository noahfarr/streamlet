import argparse
import dataclasses
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from stremax.algorithms import ACLambda, ACLambdaConfig
from stremax.environments import environment
from stremax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)
from stremax.loggers import DashboardLogger, MultiLogger, WandbLogger
from stremax.networks import heads, sparse
from stremax.optimizers import Intentional, IntentionalConfig

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
parser.add_argument(
    "--env-id",
    default="brax::halfcheetah",
    choices=[
        "brax::ant",
        "brax::halfcheetah",
        "brax::hopper",
        "brax::humanoid",
        "brax::humanoidstandup",
        "brax::inverted_double_pendulum",
        "brax::inverted_pendulum",
        "brax::pusher",
        "brax::reacher",
        "brax::swimmer",
        "brax::walker2d",
    ],
    help="Brax environment to train on.",
)
args = parser.parse_args()

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 5
env_id = args.env_id

gamma = 0.99
trace_lambda = 0.8

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

action_dim = env.action_space(env_params).shape[0]

config = ACLambdaConfig(
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

actor_optimizer = Intentional(
    cfg=IntentionalConfig(
        gamma=gamma,
        trace_lambda=trace_lambda,
        eta=0.05,
        normalize_delta=True,
    ),
)
critic_optimizer = Intentional(
    cfg=IntentionalConfig(
        gamma=gamma,
        trace_lambda=trace_lambda,
        eta=0.5,
    ),
)

agent = ACLambda(
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

group = f"intentional-AC__{env_id}__intentional"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "intentional-AC",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="intentional-AC",
            mode="online",
            group=group,
            cfg={
                "algorithm": "intentional-AC",
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                **dataclasses.asdict(config),
                "actor_optimizer": actor_optimizer.name,
                **{
                    f"actor_optimizer/{k}": v
                    for k, v in dataclasses.asdict(actor_optimizer.cfg).items()
                },
                "critic_optimizer": critic_optimizer.name,
                **{
                    f"critic_optimizer/{k}": v
                    for k, v in dataclasses.asdict(critic_optimizer.cfg).items()
                },
            },
            seed=seed,
            num_seeds=num_seeds,
        )
    )
logger = MultiLogger(loggers)

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
