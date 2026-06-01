import argparse
import dataclasses
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from streax.algorithms import RecurrentQLambda, RecurrentQLambdaConfig
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from streax.loggers import DashboardLogger, MultiLogger, WandbLogger
from streax.networks import Flatten, heads, sparse
from streax.optimizers import ObGD, ObGDConfig

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
parser.add_argument(
    "--hidden-size",
    type=int,
    default=128,
    help="GRU hidden state size.",
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

config = RecurrentQLambdaConfig(
    num_envs=1,
    trace_lambda=0.8,
    gamma=0.99,
)


class RecurrentQNetwork(nn.Module):
    """A GRU-backed recurrent Q-network following the algorithm's contract.

    The call signature is ``(carry, obs, action, reward, done) -> (carry, q)``:
    observation, previous action, reward and done all condition the recurrent
    core (R2D2-style). Any flax ``RNNCellBase`` cell works; here we use a stock
    ``nn.GRUCell``. The user owns this module entirely — the algorithm only
    threads the carry through it.
    """

    num_actions: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, carry, obs, action, reward, done):
        sparse_init = sparse(sparsity=0.9)
        x = nn.Conv(
            16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init
        )(obs)
        x = nn.LayerNorm()(x)
        x = nn.leaky_relu(x)
        x = Flatten(start_dim=-3)(x)
        x = nn.Dense(128, kernel_init=sparse_init)(x)
        x = nn.LayerNorm()(x)
        x = nn.leaky_relu(x)

        # Condition the recurrent core on the previous action, reward and done.
        context = jnp.concatenate(
            [
                jax.nn.one_hot(action, self.num_actions),
                reward[..., None],
                done[..., None].astype(jnp.float32),
            ],
            axis=-1,
        )
        x = jnp.concatenate([x, context], axis=-1)

        carry, hidden = nn.GRUCell(features=self.hidden_size)(carry, x)
        q_values = heads.DiscreteQNetwork(
            action_dim=self.num_actions, kernel_init=sparse_init
        )(hidden)
        return carry, q_values

    @nn.nowrap
    def initialize_carry(self, rng, num_envs):
        return nn.GRUCell(features=self.hidden_size).initialize_carry(
            rng, (num_envs, 1)
        )


q_network = RecurrentQNetwork(
    num_actions=num_actions, hidden_size=args.hidden_size
)

lr = args.lr if args.lr is not None else (1e-3 if args.exact else 1.0)
q_optimizer = ObGD(
    cfg=ObGDConfig(
        lr=lr,
        kappa=2.0,
        beta2=0.999,
        eps=1e-8,
        adaptive=False,
        exact=args.exact,
    ),
)

epsilon_start = 1.0
epsilon_end = 0.01
exploration_fraction = 0.2
exploration_steps = exploration_fraction * total_timesteps


def epsilon_schedule(step):
    frac = jnp.minimum(step / exploration_steps, 1.0)
    return epsilon_start + frac * (epsilon_end - epsilon_start)


agent = RecurrentQLambda(
    config,
    env,
    env_params,
    q_network,
    epsilon_schedule,
    q_optimizer,
)


init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

group = f"recurrent_q_lambda__{env_id}__obgd"

loggers = [
    DashboardLogger(
        total_timesteps=total_timesteps,
        summary={
            "Algorithm": "recurrent_q_lambda",
            "Environment": env_id,
            "Total Timesteps": f"{total_timesteps:_}",
        },
    ),
]
if args.wandb:
    loggers.append(
        WandbLogger(
            project="stremax",
            name="stream-recurrent-Q",
            mode="online",
            group=group,
            cfg={
                "algorithm": "recurrent_q_lambda",
                "env_id": env_id,
                "total_timesteps": total_timesteps,
                "hidden_size": args.hidden_size,
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
        **{k: jnp.mean(v, axis=tuple(range(1, v.ndim))) for k, v in logs.items()},
    }
    logger.log(data, step=state.step.mean(dtype=jnp.int32).item())

logger.finish()
