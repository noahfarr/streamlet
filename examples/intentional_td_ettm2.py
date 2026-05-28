import argparse
import dataclasses

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
from stremax.optimizers import Intentional, IntentionalConfig

parser = argparse.ArgumentParser()
parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
args = parser.parse_args()

total_steps = 68_000
seed = 0
num_seeds = 5
env_id = "ett::ETTm2"

gamma = 0.99
trace_lambda = 0.8
eta = 0.25
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

value_optimizer = Intentional(
    cfg=IntentionalConfig(
        gamma=gamma,
        trace_lambda=trace_lambda,
        eta=eta,
    ),
)

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

final_td_error = np.asarray(logs["value/td_error"])[:, -1000:].mean(axis=1)
print(f"final td_error (mean over last 1000 steps): {final_td_error.mean():.4f}")

group = f"intentional-TD__{env_id}__intentional"

if args.wandb:
    logger = WandbLogger(
        project="stremax",
        name="intentional-TD",
        mode="online",
        group=group,
        cfg={
            "algorithm": "intentional-TD",
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

plt.figure(figsize=(12, 4))
plt.plot(actual_returns, label="Actual Return", linewidth=3.0, color="tab:green")
plt.plot(predictions, label="Prediction", linewidth=3.0, color="tab:blue")
plt.xlim([0, 5000])
plt.xlabel("Time Step", fontsize=20)
plt.ylabel("Normalized Oil Temp.", fontsize=20)
plt.legend()

plt.figure(figsize=(12, 4))
plt.plot(actual_returns, label="Actual Return", linewidth=3.0, color="tab:green")
plt.plot(predictions, label="Prediction", linewidth=3.0, color="tab:blue")
plt.xlim([total_steps - 5000, total_steps])
plt.ylim([35, 85])
plt.xlabel("Time Step", fontsize=20)
plt.ylabel("Normalized Oil Temp.", fontsize=20)
plt.legend()
plt.show()
