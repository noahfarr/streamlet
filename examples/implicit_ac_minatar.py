import argparse
import dataclasses
import time

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from streax.algorithms import ACLambda, ACLambdaConfig
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from streax.loggers import DashboardLogger, MultiLogger, WandbLogger
from streax.networks import Flatten, sparse
from streax.optimizers import Implicit, ImplicitConfig, ObGD, ObGDConfig

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
parser.add_argument(
    "--lr", type=float, default=0.1, help="Implicit critic optimizer learning rate."
)
args = parser.parse_args()

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 5
env_id = args.env_id

env, env_params = environment.make(env_id)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env)

num_actions = env.action_space(env_params).n

config = ACLambdaConfig(
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
        nn.Dense(num_actions, kernel_init=sparse_init),
        lambda logits: distrax.Categorical(logits=logits),
    ]
)

critic_network = nn.Sequential(
    [
        network,
        nn.Dense(1, kernel_init=sparse_init),
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
critic_optimizer = Implicit(
    cfg=ImplicitConfig(gamma=0.99, trace_lambda=0.8, eta=args.lr)
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
train = jax.jit(jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=2)

group = f"ac_lambda__{env_id}__obgd-implicit"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "ac_lambda",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="implicit-AC",
            mode="online",
            group=group,
            cfg={
                "algorithm": "ac_lambda",
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
