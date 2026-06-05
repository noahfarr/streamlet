import argparse
import dataclasses
import time

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import optax
from flax.linen.initializers import orthogonal, zeros

from streax.algorithms import AVGLambda, AVGLambdaConfig
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
)
from streax.loggers import DashboardLogger, MultiLogger, WandbLogger
from streax.networks import L2Normalize
from streax.optimizers import OptaxOptimizer

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


config = AVGLambdaConfig(
    num_envs=1,
    gamma=gamma,
    alpha=alpha,
    trace_lambda=0.0,
)

# nn.Sequential passes *args to the first layer and unpacks tuple returns into
# the next, so the multi-input critic and the action-threading both work without
# a custom module. The actor's mean/log_std heads fold into one Dense(2*dim).
actor_network = nn.Sequential(
    [
        nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros),
        nn.leaky_relu,
        nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros),
        nn.leaky_relu,
        L2Normalize(),
        nn.Dense(2 * action_dim, kernel_init=orthogonal()),
        lambda out: distrax.Transformed(
            distrax.MultivariateNormalDiag(
                loc=out[..., :action_dim],
                scale_diag=jnp.exp(jnp.clip(out[..., action_dim:], -20.0, 2.0)),
            ),
            distrax.Block(distrax.Tanh(), ndims=1),
        ),
    ]
)

critic_network = nn.Sequential(
    [
        lambda obs, action: (
            nn.leaky_relu(
                nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros)(
                    jnp.concatenate([obs, action], axis=-1)
                )
            ),
            action,
        ),
        lambda features, action: (
            nn.leaky_relu(
                nn.Dense(n_hid, kernel_init=orthogonal(), bias_init=zeros)(features)
            ),
            action,
        ),
        lambda features, action: (L2Normalize()(features), action),
        lambda features, action: jnp.concatenate([features, action], axis=-1),
        nn.Dense(1, kernel_init=orthogonal()),
        lambda q_value: jnp.squeeze(q_value, axis=-1),
    ]
)

actor_optimizer = OptaxOptimizer(optax.adam(actor_lr, b1=beta1, b2=beta2, eps=eps), name="actor_optimizer")
critic_optimizer = OptaxOptimizer(optax.adam(critic_lr, b1=beta1, b2=beta2, eps=eps), name="critic_optimizer")

agent = AVGLambda(
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

group = f"AVGLambda__{env_id}__adam"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "AVGLambda",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="AVGLambda",
            mode="online",
            group=group,
            cfg={
                "algorithm": "AVGLambda",
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
