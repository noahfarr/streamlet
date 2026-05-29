import argparse
import dataclasses
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax.linen.initializers import orthogonal, zeros

from stremax.algorithms import AVG, AVGConfig
from stremax.environments import environment
from stremax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)
from stremax.loggers import DashboardLogger, MultiLogger, WandbLogger
from stremax.networks import heads
from stremax.optimizers import OptaxOptimizer

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
alpha = 0.07
actor_lr = 0.0063
critic_lr = 0.0087
beta1 = 0.0
beta2 = 0.999
eps = 1e-8
n_hid = 256

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

action_dim = env.action_space(env_params).shape[0]


def l2_normalize(x: jax.Array, eps: float = 1e-12) -> jax.Array:
    return x / jnp.maximum(jnp.linalg.norm(x, axis=-1, keepdims=True), eps)


def feature_extractor(x: jax.Array) -> jax.Array:
    x = nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros)(x)
    x = nn.leaky_relu(x)
    x = nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros)(x)
    x = nn.leaky_relu(x)
    return l2_normalize(x)


class Actor(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> object:
        features = feature_extractor(obs)
        return heads.SquashedGaussian(
            action_dim=self.action_dim, kernel_init=orthogonal()
        )(features)


class Critic(nn.Module):
    @nn.compact
    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        features = feature_extractor(jnp.concatenate([obs, action], axis=-1))
        return heads.ContinuousQNetwork(kernel_init=orthogonal())(features, action)


config = AVGConfig(
    num_envs=1,
    gamma=gamma,
    alpha=alpha,
    trace_lambda=0.0,
)

actor_network = Actor(action_dim=action_dim)
critic_network = Critic()

actor_optimizer = OptaxOptimizer(optax.adam(actor_lr, b1=beta1, b2=beta2, eps=eps))
critic_optimizer = OptaxOptimizer(optax.adam(critic_lr, b1=beta1, b2=beta2, eps=eps))

agent = AVG(
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

group = f"AVG__{env_id}__adam"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "AVG",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="AVG",
            mode="online",
            group=group,
            cfg={
                "algorithm": "AVG",
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                **dataclasses.asdict(config),
                "actor_optimizer": "adam",
                "critic_optimizer": "adam",
                "actor_lr": actor_lr,
                "critic_lr": critic_lr,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "n_hid": n_hid,
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
