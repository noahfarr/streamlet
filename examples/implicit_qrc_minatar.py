import argparse
import dataclasses
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax

from streax.algorithms import QRCLambda, QRCLambdaConfig
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from streax.loggers import DashboardLogger, MultiLogger, WandbLogger
from streax.networks import Flatten, sparse
from streax.optimizers import Implicit, ImplicitConfig, OptaxOptimizer

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
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
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

num_actions = env.action_space(env_params).n

config = QRCLambdaConfig(
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
        nn.Dense(num_actions, kernel_init=sparse_init),
    ]
)
h_network = nn.Sequential(
    [
        backbone,
        nn.Dense(num_actions, kernel_init=sparse_init),
    ]
)

eta = 0.5
q_optimizer = Implicit(
    cfg=ImplicitConfig(gamma=gamma, trace_lambda=trace_lambda, eta=eta),
    name="q_optimizer",
)
h_lr = 0.1 * 1e-4
h_optimizer = OptaxOptimizer(tx=optax.sgd(h_lr), name="h_optimizer")

epsilon_start = 1.0
epsilon_end = 0.01
exploration_fraction = 0.2
epsilon_schedule = optax.linear_schedule(
    epsilon_start, epsilon_end, int(total_timesteps * exploration_fraction)
)

agent = QRCLambda(
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
train = jax.jit(jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=2)

group = f"implicit_qrc__{env_id}__implicit"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "implicit_qrc",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="ImplicitQRCLambda",
            mode="online",
            group=group,
            cfg={
                "algorithm": "implicit_qrc",
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                **dataclasses.asdict(config),
                "q_optimizer": type(q_optimizer).__name__.lower(),
                "h_optimizer": type(h_optimizer).__name__.lower(),
                "eta": eta,
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
