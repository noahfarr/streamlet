import argparse
import dataclasses
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax

from stremax.algorithms import QRC, QRCConfig
from stremax.environments import environment
from stremax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)
from stremax.loggers import DashboardLogger, MultiLogger, WandbLogger
from stremax.networks import Flatten, heads, sparse
from stremax.optimizers import OptaxOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
args = parser.parse_args()

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 5
env_id = "gymnax::Breakout-MinAtar"

gamma = 0.99
trace_lambda = 0.8

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

num_actions = env.action_space(env_params).n

config = QRCConfig(
    num_envs=1,
    gamma=gamma,
    trace_lambda=trace_lambda,
    gradient_correction=True,
    regularization_coefficient=1.0,
    unroll=2,
)

sparse_init = sparse(sparsity=0.9)
backbone = nn.Sequential(
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
        backbone,
        heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=sparse_init),
    ]
)
h_network = nn.Sequential(
    [
        backbone,
        heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=sparse_init),
    ]
)

q_lr = 1e-4
h_lr = 0.1 * q_lr
q_optimizer = OptaxOptimizer(tx=optax.sgd(q_lr), name="q_optimizer")
h_optimizer = OptaxOptimizer(tx=optax.sgd(h_lr), name="h_optimizer")

epsilon_start = 1.0
epsilon_end = 0.01
exploration_fraction = 0.2
epsilon_schedule = optax.linear_schedule(
    epsilon_start, epsilon_end, int(total_timesteps * exploration_fraction)
)

agent = QRC(
    cfg=config,
    env=env,
    env_params=env_params,
    q_network=q_network,
    h_network=h_network,
    q_optimizer=q_optimizer,
    h_optimizer=h_optimizer,
    epsilon_schedule=epsilon_schedule,
)


init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

group = f"QRC__{env_id}__sgd"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "QRC",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="QRC",
            mode="online",
            group=group,
            cfg={
                "algorithm": "QRC",
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                **dataclasses.asdict(config),
                "q_optimizer": q_optimizer.name,
                "h_optimizer": h_optimizer.name,
                "q_lr": q_lr,
                "h_lr": h_lr,
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
