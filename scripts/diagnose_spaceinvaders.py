import argparse

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import numpy as np

from streax.algorithms import QLambda, QLambdaConfig
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from streax.networks import Flatten, heads, sparse
from streax.optimizers import Measured, MeasuredConfig, MeasuredMode

p = argparse.ArgumentParser()
p.add_argument("--env-id", default="gymnax::SpaceInvaders-MinAtar")
p.add_argument("--total", type=int, default=1_500_000)
p.add_argument("--epoch", type=int, default=50_000)
p.add_argument("--eta", type=float, default=0.5)
p.add_argument("--nu", type=float, default=0.01)
p.add_argument("--beta", type=float, default=0.999)
p.add_argument("--huber", action="store_true")
p.add_argument("--rmsprop", action="store_true")
p.add_argument("--mode", default="operator", choices=["operator", "frobenius"])
args = p.parse_args()

gamma, trace_lambda = 0.99, 0.8

env, env_params = environment.make(args.env_id)
env = StickyActionWrapper(env)
env = RecordEpisodeStatistics(env)
env = NormalizeObservationWrapper(env)
env = NormalizeRewardWrapper(env, gamma=gamma)
num_actions = env.action_space(env_params).n

sparse_init = sparse(sparsity=0.9)
backbone = nn.Sequential([
    nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init),
    nn.LayerNorm(), nn.leaky_relu, Flatten(start_dim=-3),
    nn.Dense(128, kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
])
q_network = nn.Sequential([
    backbone, heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=sparse_init),
])

total_timesteps = 5_000_000
exploration_steps = 0.2 * total_timesteps


def epsilon_schedule(step):
    frac = jnp.minimum(step / exploration_steps, 1.0)
    return 1.0 + frac * (0.01 - 1.0)


config = QLambdaConfig(num_envs=1, gamma=gamma, trace_lambda=trace_lambda)
optimizer = Measured(cfg=MeasuredConfig(eta=args.eta, nu=args.nu, beta=args.beta, huber=args.huber, precondition=args.rmsprop, mode=MeasuredMode(args.mode)))
agent = QLambda(config, env, env_params, q_network, epsilon_schedule, optimizer)

train = lox.spool(agent.train)
key = jax.random.key(0)
key, init_key = jax.random.split(key)
state = agent.init(init_key)

keys = [
    "measured/m_hat", "measured/s_hat", "measured/y_hat",
    "measured/step_size", "measured/expansive_fraction",
    "q_network/q_value", "q_network/td_error", "q_trace/trace_norm",
]
print(f"\n=== {args.env_id}  measured(eta={args.eta},nu={args.nu},beta={args.beta})  single seed ===")
hdr = f"{'Mstep':>7}{'eps':>6}{'m_hat E[X]':>12}{'s_hat E[X2]':>12}{'y_hat':>11}{'alpha':>10}{'gate%':>7}{'Qval':>11}{'tderr':>11}{'trace_nrm':>11}"
print(hdr); print("-" * len(hdr))

n_epochs = args.total // args.epoch
for e in range(n_epochs):
    key, tk = jax.random.split(key)
    state, logs = train(tk, state, args.epoch)
    jax.block_until_ready(state)
    v = {k: float(np.asarray(logs[k]).mean()) for k in keys}
    step = (e + 1) * args.epoch
    eps = float(epsilon_schedule(step))
    print(f"{step/1e6:>7.2f}{eps:>6.2f}"
          f"{v['measured/m_hat']:>12.2e}{v['measured/s_hat']:>12.2e}{v['measured/y_hat']:>11.2e}"
          f"{v['measured/step_size']:>10.2e}{v['measured/expansive_fraction']*100:>7.0f}"
          f"{v['q_network/q_value']:>11.2e}{v['q_network/td_error']:>11.2e}{v['q_trace/trace_norm']:>11.2e}")
