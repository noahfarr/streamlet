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
p.add_argument("--total", type=int, default=1_000_000)
p.add_argument("--epoch", type=int, default=100_000)
args = p.parse_args()

gamma, trace_lambda = 0.99, 0.8


def build_env():
    env, env_params = environment.make(args.env_id)
    env = StickyActionWrapper(env)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=gamma)
    return env, env_params


def make_network(num_actions):
    si = sparse(sparsity=0.9)
    backbone = nn.Sequential([
        nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=si),
        nn.LayerNorm(), nn.leaky_relu, Flatten(start_dim=-3),
        nn.Dense(128, kernel_init=si), nn.LayerNorm(), nn.leaky_relu,
    ])
    return nn.Sequential([backbone, heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=si)])


exploration_steps = 0.2 * 5_000_000


def epsilon_schedule(step):
    return 1.0 + jnp.minimum(step / exploration_steps, 1.0) * (0.01 - 1.0)


configs = {
    "obgd(lr=1)": lambda: ObGD(cfg=ObGDConfig(lr=1.0, kappa=2.0)),
    "measured(0.5)": lambda: Measured(cfg=MeasuredConfig(eta=0.5)),
    "measured(0.5)+P+H": lambda: Measured(
        cfg=MeasuredConfig(eta=0.5, precondition=True, huber=True)
    ),
    "adam": lambda: AdaptiveQ(cfg=AdaptiveQConfig(gamma=gamma, trace_lambda=trace_lambda)),
}

config = QLambdaConfig(num_envs=1, gamma=gamma, trace_lambda=trace_lambda)
n_epochs = args.total // args.epoch
results = {}

for name, make_opt in configs.items():
    env, env_params = build_env()
    num_actions = env.action_space(env_params).n
    agent = QLambda(config, env, env_params, make_network(num_actions), epsilon_schedule, make_opt())
    train = lox.spool(agent.train)
    key = jax.random.key(0)
    key, ik = jax.random.split(key)
    state = agent.init(ik)
    rets, qs = [], []
    for e in range(n_epochs):
        key, tk = jax.random.split(key)
        state, logs = train(tk, state, args.epoch)
        jax.block_until_ready(state)
        mask = np.asarray(logs["returned_episode"]).astype(bool)
        r = np.asarray(logs["returned_episode_returns"])
        rets.append(float(r[mask].mean()) if mask.any() else float("nan"))
        qs.append(float(np.asarray(logs["q_network/q_value"]).mean()))
    results[name] = (rets, qs)

steps = [f"{(e+1)*args.epoch/1e6:.1f}M" for e in range(n_epochs)]
print(f"\n=== {args.env_id}  single seed  (episode return per {args.epoch//1000}k-step epoch) ===\n")
print(f"{'optimizer':<20}" + "".join(f"{s:>9}" for s in steps))
print("-" * (20 + 9 * n_epochs))
for name, (rets, _) in results.items():
    print(f"{name:<20}" + "".join(f"{r:>9.2f}" for r in rets))
print(f"\n=== mean Q-value (stability check) ===\n")
print(f"{'optimizer':<20}" + "".join(f"{s:>9}" for s in steps))
print("-" * (20 + 9 * n_epochs))
for name, (_, qs) in results.items():
    print(f"{name:<20}" + "".join(f"{q:>9.1e}" for q in qs))
