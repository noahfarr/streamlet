import argparse
import dataclasses
import time

import flax.linen as nn
import gymnasium
import jax
import jax.numpy as jnp
import lox
import pufferlib.emulation
import pufferlib.vector

from streax.algorithms import QLambda, QLambdaConfig
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from streax.loggers import DashboardLogger, MultiLogger, WandbLogger
from streax.networks import Flatten, sparse
from streax.optimizers import ObGD, ObGDConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb", action="store_true", help="Enable Weights & Biases logging."
)
parser.add_argument(
    "--lr",
    type=float,
    default=None,
    help="ObGD base step size. Default: 1.0 for the bound, 1e-3 for --exact "
    "(lr=1.0 diverges with --exact).",
)
args = parser.parse_args()

total_timesteps = 500_000
num_epochs = 10
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 10
suite = "pufferlib"
env_id = "CartPole-v1"
num_envs = 1

env, env_params = environment.make(
    f"{suite}::{env_id}",
    env_creator=pufferlib.emulation.GymnasiumPufferEnv,
    env_kwargs={"env_creator": lambda: gymnasium.make(env_id)},
    batch_shape=(num_seeds, num_envs),
    backend=pufferlib.vector.Serial,
    num_workers=1,
)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env)

num_actions = env.action_space(env_params).n

config = QLambdaConfig(
    num_envs=num_envs,
    trace_lambda=0.8,
    gamma=0.99,
)

sparse_init = sparse(sparsity=0.9)
q_network = nn.Sequential(
    [
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        nn.Dense(num_actions, kernel_init=sparse_init),
    ]
)

q_optimizer = ObGD(
    cfg=ObGDConfig(
        lr=1.0,
        kappa=2.0,
        beta2=0.999,
        eps=1e-8,
        adaptive=False,
    ),
    name="q_optimizer",
)

epsilon_start = 1.0
epsilon_end = 0.01
exploration_fraction = 0.2
exploration_steps = exploration_fraction * total_timesteps


def epsilon_schedule(step):
    frac = jnp.minimum(step / exploration_steps, 1.0)
    return epsilon_start + frac * (epsilon_end - epsilon_start)


agent = QLambda(
    config,
    env,
    env_params,
    q_network,
    epsilon_schedule,
    q_optimizer,
)


init = jax.vmap(agent.init)
train = jax.jit(
    jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=2
)

group = f"q_lambda__{env_id}__obgd"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "q_lambda",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="stream-Q",
            mode="online",
            group=group,
            cfg={
                "algorithm": "q_lambda",
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                **dataclasses.asdict(config),
                "q_optimizer": type(q_optimizer).__name__.lower(),
                **{
                    f"q_optimizer/{k}": v
                    for k, v in dataclasses.asdict(q_optimizer.cfg).items()
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
state = jax.tree.map(lambda x: jax.lax.convert_element_type(x, x.dtype), state)

for i in range(num_epochs):
    start = time.perf_counter()
    key, train_key = jax.random.split(key)
    state, logs = train(jax.random.split(train_key, num_seeds), state, num_steps)
    jax.block_until_ready(state)
    end = time.perf_counter()

    SPS = int(num_steps * num_seeds / (end - start))

    mask = logs.pop("returned_episode")
    episode_returns = jnp.where(mask, logs.pop("returned_episode_returns"), jnp.nan)
    episode_lengths = jnp.where(mask, logs.pop("returned_episode_lengths"), jnp.nan)
    discounted_episode_returns = jnp.where(
        mask, logs.pop("returned_discounted_episode_returns"), jnp.nan
    )

    sps = jnp.full((num_seeds, num_steps), jnp.nan).at[:, -1].set(SPS)

    data = {
        "training/SPS": sps,
        "training/episode_returns": episode_returns,
        "training/episode_lengths": episode_lengths,
        "training/discounted_episode_returns": discounted_episode_returns,
        **logs,
    }
    steps = i * num_steps + jnp.arange(num_steps)
    logger.log(data, steps)

logger.finish()
