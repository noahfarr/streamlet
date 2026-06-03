import argparse
import dataclasses
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

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
from streax.optimizers import Calibrated, CalibratedConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wandb", action="store_true", help="Enable Weights & Biases logging."
)
parser.add_argument(
    "--env-id",
    default="gymnax::Breakout-MinAtar",
    choices=[
        "gymnax::Asterix-MinAtar",
        "gymnax::Breakout-MinAtar",
        "gymnax::Freeway-MinAtar",
        "gymnax::SpaceInvaders-MinAtar",
    ],
    help="MinAtar environment to train on.",
)
parser.add_argument(
    "--precondition",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Diagonal RMSProp preconditioner on the update direction.",
)
parser.add_argument(
    "--huber",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Clip the TD error to +/- huber_delta before the update.",
)
parser.add_argument(
    "--nu",
    type=float,
    default=1.0,
    help="Weight on the target-variance term E[delta^2 ||z||^2] in the denominator.",
)
parser.add_argument(
    "--beta",
    type=float,
    default=0.999,
    help="EMA decay for the moment estimates in the denominator.",
)
args = parser.parse_args()

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 10
env_id = args.env_id

gamma = 0.99
trace_lambda = 0.8

env, env_params = environment.make(env_id)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

num_actions = env.action_space(env_params).n

config = QLambdaConfig(
    num_envs=1,
    trace_lambda=trace_lambda,
    gamma=gamma,
)

sparse_init = sparse(sparsity=0.9)
q_network = nn.Sequential(
    [
        nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        Flatten(start_dim=-3),
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        nn.Dense(num_actions, kernel_init=sparse_init),
    ]
)

q_optimizer = Calibrated(
    cfg=CalibratedConfig(
        precondition=args.precondition,
        huber=args.huber,
        nu=args.nu,
        beta=args.beta,
    )
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
train = jax.jit(jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=2)

group = f"q_lambda__{env_id}__calibrated"

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
            name="calibrated-Q",
            mode="online",
            group=group,
            cfg={
                "algorithm": "q_lambda",
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                **dataclasses.asdict(config),
                "optimizer": q_optimizer.name,
                **{
                    f"optimizer/{k}": v
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

    SPS = int(num_steps / (end - start))

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
