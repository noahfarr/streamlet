import argparse
from pathlib import Path

import flax.linen as nn
import jax
import lox
import matplotlib.pyplot as plt
import numpy as np

from streax.algorithms import TDLambda, TDLambdaConfig
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    ObservationTracesWrapper,
    RecordEpisodeStatistics,
)
from streax.networks import sparse
from streax.optimizers import Implicit, ImplicitConfig

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env-id",
    default="ett::ETTm2",
    choices=[
        "ett::ETTh1",
        "ett::ETTh2",
        "ett::ETTm1",
        "ett::ETTm2",
    ],
    help="ETT dataset to train on.",
)
parser.add_argument(
    "--lr", type=float, default=1.0, help="Implicit optimizer learning rate."
)
args = parser.parse_args()

total_steps = 68_000
seed = 0
num_seeds = 30
env_id = args.env_id

gamma = 0.99
trace_lambda = 0.8
lr = args.lr
beta = 0.999


def build_env():
    env, env_params = environment.make(env_id)
    env = RecordEpisodeStatistics(env, gamma=gamma)
    env = ObservationTracesWrapper(env, beta=beta)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=gamma)
    return env, env_params


def value_network():
    sparse_init = sparse(sparsity=0.9)
    return nn.Sequential(
        [
            nn.Dense(128, kernel_init=sparse_init),
            nn.LayerNorm(),
            nn.leaky_relu,
            nn.Dense(128, kernel_init=sparse_init),
            nn.LayerNorm(),
            nn.leaky_relu,
            nn.Dense(1, kernel_init=sparse_init),
        ]
    )


env, env_params = build_env()
config = TDLambdaConfig(num_envs=1, gamma=gamma, trace_lambda=trace_lambda)
value_optimizer = Implicit(
    cfg=ImplicitConfig(gamma=gamma, trace_lambda=trace_lambda, eta=lr)
)
agent = TDLambda(config, env, env_params, value_network(), value_optimizer)

init = jax.vmap(agent.init)
train = jax.jit(jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=2)

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))
state = jax.tree.map(lambda x: jax.lax.convert_element_type(x, x.dtype), state)

key, train_key = jax.random.split(key)
state, logs = train(jax.random.split(train_key, num_seeds), state, total_steps)
jax.block_until_ready(state)

std = np.asarray(logs["normalize_reward/std"]).reshape(num_seeds, total_steps)
predictions = np.asarray(logs["value/value"]).reshape(num_seeds, total_steps) * std
cumulants = np.asarray(logs["value/cumulant"]).reshape(num_seeds, total_steps) * std

actual_returns = np.zeros((num_seeds, total_steps))
return_t = np.zeros(num_seeds)
for t in reversed(range(total_steps)):
    return_t = return_t * gamma + cumulants[:, t]
    actual_returns[:, t] = return_t

plot_dir = Path("plots") / env_id / "implicit-TD"
plot_dir.mkdir(parents=True, exist_ok=True)


def plot_window(ax, steps):
    mean = predictions.mean(axis=0)
    sem = predictions.std(axis=0, ddof=1) / np.sqrt(predictions.shape[0])
    ci = 1.96 * sem
    ax.plot(
        steps, actual_returns[0, steps], label="Actual Return",
        linewidth=3.0, color="tab:green",
    )
    ax.plot(steps, mean[steps], label="Prediction", linewidth=3.0, color="tab:blue")
    ax.fill_between(
        steps, (mean - ci)[steps], (mean + ci)[steps], color="tab:blue", alpha=0.3
    )
    ax.set_xlabel("Time Step", fontsize=20)
    ax.set_ylabel("Normalized Oil Temp.", fontsize=20)
    ax.legend()


fig, ax = plt.subplots(figsize=(12, 4))
plot_window(ax, np.arange(0, 5000))
fig.savefig(plot_dir / "start.png", dpi=150, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
plot_window(ax, np.arange(total_steps - 5000, total_steps))
ax.set_ylim([35, 85])
fig.savefig(plot_dir / "end.png", dpi=150, bbox_inches="tight")
plt.close(fig)

print(f"Saved plots to {plot_dir}")
