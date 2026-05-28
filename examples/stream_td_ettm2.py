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
from stremax.networks import heads, sparse
from stremax.optimizers import ObGD, ObGDConfig

total_steps = 68_000
seed = 0
num_seeds = 1
env_id = "ett::ETTm2"

gamma = 0.99
trace_lambda = 0.8
lr = 1.0
kappa = 2.0
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

value_optimizer = ObGD(cfg=ObGDConfig(lr=lr, kappa=kappa))

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

print(f"final td_error (mean over last 1000 steps): "
      f"{np.asarray(logs['value/td_error'][0])[-1000:].mean():.4f}")

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
plt.savefig("td_ettm2_start.pdf", bbox_inches="tight")

plt.figure(figsize=(12, 4))
plt.plot(actual_returns, label="Actual Return", linewidth=3.0, color="tab:green")
plt.plot(predictions, label="Prediction", linewidth=3.0, color="tab:blue")
plt.xlim([total_steps - 5000, total_steps])
plt.ylim([35, 85])
plt.xlabel("Time Step", fontsize=20)
plt.ylabel("Normalized Oil Temp.", fontsize=20)
plt.legend()
plt.savefig("td_ettm2_end.pdf", bbox_inches="tight")
