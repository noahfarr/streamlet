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
from streax.optimizers import (
    Adaptive,
    AdaptiveConfig,
    Implicit,
    ImplicitConfig,
    Intentional,
    IntentionalConfig,
    Calibrated,
    CalibratedConfig,
    ObGD,
    ObGDConfig,
)

total_timesteps = 5_000_000
num_epochs = 100
num_steps = total_timesteps // num_epochs
seed = 0
num_seeds = 5

gamma = 0.99
trace_lambda = 0.8

ENV_IDS = [
    "gymnax::Asterix-MinAtar",
    "gymnax::Breakout-MinAtar",
    "gymnax::Freeway-MinAtar",
    "gymnax::SpaceInvaders-MinAtar",
]

# Each entry builds a fresh optimizer instance with its tuned hyperparameters.
OPTIMIZERS = {
    "calibrated": lambda: Calibrated(cfg=CalibratedConfig(), name="optimizer"),
    "implicit": lambda: Implicit(
        cfg=ImplicitConfig(gamma=gamma, trace_lambda=trace_lambda, eta=0.25),
        name="optimizer",
    ),
    "intentional": lambda: Intentional(
        cfg=IntentionalConfig(gamma=gamma, trace_lambda=trace_lambda, eta=0.25),
        name="optimizer",
    ),
    "adaptive": lambda: Adaptive(
        cfg=AdaptiveConfig(
            gamma=gamma,
            trace_lambda=trace_lambda,
            eta=4.6e-4,
            eps=0.1,
            clip=1.0,
        ),
        name="optimizer",
    ),
    "obgd": lambda: ObGD(
        cfg=ObGDConfig(lr=1.0, kappa=2.0, beta2=0.999, eps=1e-8, adaptive=False),
        name="optimizer",
    ),
}

epsilon_start = 1.0
epsilon_end = 0.01
exploration_fraction = 0.2
exploration_steps = exploration_fraction * total_timesteps


def epsilon_schedule(step):
    frac = jnp.minimum(step / exploration_steps, 1.0)
    return epsilon_start + frac * (epsilon_end - epsilon_start)


def run(env_id, opt_name, q_optimizer, use_wandb):
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
            nn.Conv(
                16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init
            ),
            nn.LayerNorm(),
            nn.leaky_relu,
            Flatten(start_dim=-3),
            nn.Dense(128, kernel_init=sparse_init),
            nn.LayerNorm(),
            nn.leaky_relu,
            nn.Dense(num_actions, kernel_init=sparse_init),
        ]
    )

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

    group = f"q_lambda__{env_id}__{opt_name}"

    loggers = [
        DashboardLogger(
            total_timesteps=total_timesteps,
            summary={
                "Algorithm": "q_lambda",
                "Environment": env_id,
                "Optimizer": opt_name,
                "Total Timesteps": f"{total_timesteps:_}",
            },
        ),
    ]
    if use_wandb:
        loggers.append(
            WandbLogger(
                project="stremax",
                name=f"{opt_name}-Q",
                mode="online",
                group=group,
                cfg={
                    "algorithm": "q_lambda",
                    "env_id": env_id,
                    "total_timesteps": total_timesteps,
                    **dataclasses.asdict(config),
                    "optimizer": opt_name,
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


def main():
    parser = argparse.ArgumentParser(
        description="Run stream-Q with every optimizer across every MinAtar env."
    )
    parser.add_argument(
        "--wandb", action="store_true", help="Enable Weights & Biases logging."
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=list(OPTIMIZERS),
        choices=list(OPTIMIZERS),
        help="Subset of optimizers to run (default: all).",
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=ENV_IDS,
        choices=ENV_IDS,
        help="Subset of MinAtar environments to run (default: all).",
    )
    parser.add_argument(
        "--exact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use ObGD's exact (JVP) effective step size instead of the |z|_1 bound.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="ObGD base step size. Default: 1.0 for the bound, 1e-3 for --exact "
        "(lr=1.0 diverges with --exact).",
    )
    args = parser.parse_args()

    obgd_lr = args.lr if args.lr is not None else (1e-3 if args.exact else 1.0)
    OPTIMIZERS["obgd"] = lambda: ObGD(
        cfg=ObGDConfig(
            lr=obgd_lr,
            kappa=2.0,
            beta2=0.999,
            eps=1e-8,
            adaptive=False,
            exact=args.exact,
        ),
        name="optimizer",
    )

    for opt_name in args.optimizers:
        for env_id in args.envs:
            print(f"=== Running {opt_name} on {env_id} ===", flush=True)
            run(env_id, opt_name, OPTIMIZERS[opt_name](), args.wandb)


if __name__ == "__main__":
    main()
