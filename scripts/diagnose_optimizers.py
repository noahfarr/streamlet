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
from streax.optimizers import (
    AdaptiveQ,
    AdaptiveQConfig,
    Measured,
    MeasuredConfig,
    ObGD,
    ObGDConfig,
)

p = argparse.ArgumentParser()
p.add_argument("--env-id", default="gymnax::SpaceInvaders-MinAtar")
p.add_argument("--optimizer", default="adaptive", choices=["adaptive", "measured", "obgd"])
p.add_argument("--total", type=int, default=500_000)
p.add_argument("--epoch", type=int, default=50_000)
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

exploration_steps = 0.2 * 5_000_000


def epsilon_schedule(step):
    return 1.0 + jnp.minimum(step / exploration_steps, 1.0) * (0.01 - 1.0)


optimizers = {
    "adaptive": AdaptiveQ(cfg=AdaptiveQConfig(gamma=gamma, trace_lambda=trace_lambda)),
    "measured": Measured(cfg=MeasuredConfig(eta=0.5)),
    "obgd": ObGD(cfg=ObGDConfig(lr=1.0, kappa=2.0)),
}
optimizer = optimizers[args.optimizer]

config = QLambdaConfig(num_envs=1, gamma=gamma, trace_lambda=trace_lambda)
agent = QLambda(config, env, env_params, q_network, epsilon_schedule, optimizer)

train = lox.spool(agent.train)
key = jax.random.key(0)
key, init_key = jax.random.split(key)
state = agent.init(init_key)

print(f"\n=== {args.env_id}  optimizer={args.optimizer} ({optimizer.name})  single seed ===", flush=True)
hdr = f"{'Mstep':>7}{'eps':>6}{'Qval':>12}{'tderr':>12}{'trace_nrm':>12}{'step_size':>12}"
print(hdr, flush=True)
print("-" * len(hdr), flush=True)

n_epochs = args.total // args.epoch
for e in range(n_epochs):
    key, tk = jax.random.split(key)
    state, logs = train(tk, state, args.epoch)
    jax.block_until_ready(state)

    def m(k):
        return float(np.asarray(logs[k]).mean()) if k in logs else float("nan")

    step = (e + 1) * args.epoch
    eps = float(epsilon_schedule(step))
    print(f"{step/1e6:>7.2f}{eps:>6.2f}"
          f"{m('q_network/q_value'):>12.3e}{m('q_network/td_error'):>12.3e}"
          f"{m('q_trace/trace_norm'):>12.3e}{m(f'{optimizer.name}/step_size'):>12.3e}",
          flush=True)
