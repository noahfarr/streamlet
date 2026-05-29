import argparse
import dataclasses
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import matplotlib.pyplot as plt
import numpy as np

from stremax.algorithms import StreamTD, StreamTDConfig
from stremax.environments import environment
from stremax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    ObservationTracesWrapper,
    RecordEpisodeStatistics,
)
from stremax.loggers import WandbLogger
from stremax.networks import heads, sparse
from stremax.optimizers import Implicit, ImplicitConfig

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
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
num_seeds = 5
env_id = args.env_id

gamma = 0.99
trace_lambda = 0.8
lr = args.lr
beta = 0.999

env, env_params = environment.make(env_id)
env = RecordEpisodeStatistics(env, gamma=gamma)
env = ObservationTracesWrapper(env, beta=beta)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)

n_obs = env.observation_space(env_params).shape[0]

config = StreamTDConfig(
    num_envs=1,
    gamma=gamma,
    trace_lambda=trace_lambda,
)

sparse_init = sparse(sparsity=0.9)
value_network = nn.Sequential(
    [
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        heads.VNetwork(kernel_init=sparse_init),
    ]
)

value_optimizer = Implicit(cfg=ImplicitConfig(lr=lr))

agent = StreamTD(
    config,
    env,
    env_params,
    value_network,
    value_optimizer,
)

init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(seed)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

key, train_key = jax.random.split(key)
state, logs = train(jax.random.split(train_key, num_seeds), state, total_steps)
jax.block_until_ready(state)

std = np.asarray(logs["normalize_reward/std"][0]).squeeze()
predictions = np.asarray(logs["value/value"][0]) * std
cumulants = np.asarray(logs["value/cumulant"][0]) * std

td_error = np.asarray(logs["value/td_error"])[:, -1000:]
final_td_error = td_error.mean(axis=1)
final_td_rms = np.sqrt(np.square(td_error).mean(axis=1))
final_td_mae = np.abs(td_error).mean(axis=1)
print(
    "final td_error over last 1000 steps -- "
    f"signed-mean: {final_td_error.mean():.4f}  "
    f"RMS: {final_td_rms.mean():.4f}  "
    f"MAE: {final_td_mae.mean():.4f}"
)

group = f"implicit-TD__{env_id}__implicit"

if args.wandb:
    logger = WandbLogger(
        project="stremax",
        name="implicit-TD",
        mode="online",
        group=group,
        cfg={
            "algorithm": "implicit-TD",
            "env_id": env_id,
            "total_steps": total_steps,
            **dataclasses.asdict(config),
            "optimizer": value_optimizer.name,
            **{
                f"optimizer/{k}": v
                for k, v in dataclasses.asdict(value_optimizer.cfg).items()
            },
        },
        seed=seed,
        num_seeds=num_seeds,
    )
    logger.log({"value/td_error": final_td_error}, step=total_steps)
    logger.finish()

actual_returns = np.zeros(total_steps)
return_t = 0.0
for t in reversed(range(total_steps)):
    return_t = return_t * gamma + cumulants[t]
    actual_returns[t] = return_t

plot_dir = Path("plots") / env_id / "implicit-TD"
plot_dir.mkdir(parents=True, exist_ok=True)


def crop_to_both(lo, hi):
    """Set y-limits so both series are in view over the window [lo, hi)."""
    window = np.concatenate([actual_returns[lo:hi], predictions[lo:hi]])
    margin = 0.05 * (window.max() - window.min())
    plt.ylim([window.min() - margin, window.max() + margin])


plt.figure(figsize=(12, 4))
plt.plot(actual_returns, label="Actual Return", linewidth=3.0, color="tab:green")
plt.plot(predictions, label="Prediction", linewidth=3.0, color="tab:blue")
plt.xlim([0, 5000])
crop_to_both(0, 5000)
plt.xlabel("Time Step", fontsize=20)
plt.ylabel("Normalized Oil Temp.", fontsize=20)
plt.legend()
plt.savefig(plot_dir / "start.png", dpi=150, bbox_inches="tight")

plt.figure(figsize=(12, 4))
plt.plot(actual_returns, label="Actual Return", linewidth=3.0, color="tab:green")
plt.plot(predictions, label="Prediction", linewidth=3.0, color="tab:blue")
plt.xlim([total_steps - 5000, total_steps])
crop_to_both(total_steps - 5000, total_steps)
plt.xlabel("Time Step", fontsize=20)
plt.ylabel("Normalized Oil Temp.", fontsize=20)
plt.legend()
plt.savefig(plot_dir / "end.png", dpi=150, bbox_inches="tight")

print(f"Saved plots to {plot_dir}")
