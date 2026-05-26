# Strelax

**Streaming reinforcement learning in JAX.**

Strelax implements *streaming* deep RL algorithms — agents that learn online from a
single stream of experience, one transition at a time, with no replay buffer and a
batch size of one. The implementations are fully functional, `jit`-compatible, and
designed to `vmap` over random seeds for fast, reproducible experiments.

> The name is a portmanteau of **stream** and **[JAX](https://github.com/jax-ml/jax)**.

## Features

- **Stream Q** — streaming Q-learning with eligibility traces and ε-greedy exploration.
- **Stream AC** — streaming actor–critic with entropy regularization.
- **Eligibility-trace optimizers** — `OBGD` (overshooting-bounded gradient descent)
  and `IntentionalOptimizer`, plus an `optax` wrapper for standard optimizers.
- **Environments** — a single `make("namespace::env_id")` entry point over
  [`gymnax`](https://github.com/RobertTLange/gymnax) and
  [`brax`](https://github.com/google/brax).
- **Wrappers** — observation/reward normalization, episode-statistics recording,
  and sticky actions.
- **Networks** — Flax modules for discrete Q-values, value functions, categorical and
  Gaussian policy heads, and sparse weight initialization.
- **Logging** — structured, in-graph logging via [`lox`](https://github.com/huterguier/lox).

## Installation

Strelax uses [`uv`](https://github.com/astral-sh/uv) and requires Python ≥ 3.12.

```bash
git clone https://github.com/noahfarr/strelax.git
cd strelax
uv sync
```

This installs JAX with CUDA 12 support on Linux and CPU/Metal JAX on macOS. To add
Strelax to an existing project:

```bash
uv add git+https://github.com/noahfarr/strelax.git
```

## Quickstart

Train a streaming Q-learning agent on MinAtar Breakout:

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
import lox

from strelax.algorithms import StreamQ, StreamQConfig
from strelax.environments import environment
from strelax.networks import Flatten, heads, sparse
from strelax.optimizers import OBGD, OBGDConfig

env, env_params = environment.make("gymnax::Breakout-MinAtar")
num_actions = env.action_space(env_params).n

config = StreamQConfig(num_envs=1, gamma=0.99, trace_lambda=0.8)

sparse_init = sparse(sparsity=0.9)
q_network = nn.Sequential([
    nn.Conv(16, (3, 3), padding="VALID", kernel_init=sparse_init),
    nn.LayerNorm(), nn.leaky_relu,
    Flatten(start_dim=-3),
    nn.Dense(128, kernel_init=sparse_init),
    nn.LayerNorm(), nn.leaky_relu,
    heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=sparse_init),
])

q_optimizer = OBGD(OBGDConfig(lr=1.0, kappa=2.0))
epsilon_schedule = lambda step: jnp.maximum(1.0 - step / 1e6, 0.01)

agent = StreamQ(config, env, env_params, q_network, epsilon_schedule, q_optimizer)

# Vectorize over seeds; lox.spool collects logged metrics from inside the scan.
num_seeds = 1
init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(0)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))
state, logs = train(jax.random.split(key, num_seeds), state, num_steps=100_000)
```

Every algorithm exposes the same interface:

```python
state = agent.init(key)                      # initialize parameters, traces, env state
state = agent.warmup(key, state, num_steps)  # optional random pre-fill
state = agent.train(key, state, num_steps)   # streaming updates
state = agent.evaluate(key, state, num_steps)
```

## Examples

Runnable scripts live in [`examples/`](examples/):

| Script | Algorithm | Optimizer | Environment |
| --- | --- | --- | --- |
| `stream_q_minatar.py` | Stream Q | OBGD | MinAtar (gymnax) |
| `stream_ac_minatar.py` | Stream AC | OBGD | MinAtar (gymnax) |
| `intentional_q_minatar.py` | Stream Q | Intentional | MinAtar (gymnax) |
| `intentional_ac_brax.py` | Stream AC | Intentional | HalfCheetah (brax) |

Run one with:

```bash
uv run python examples/stream_q_minatar.py
```

## Project layout

```
strelax/
├── algorithms/      # StreamQ, StreamAC
├── optimizers/      # OBGD, IntentionalOptimizer, optax wrapper
├── environments/    # gymnax & brax adapters + wrappers
├── networks/        # heads, layers, sparse initialization
└── utils/           # Timestep, Transition, helpers
```

## License

[MIT](LICENSE) © 2026 Noah Farr
