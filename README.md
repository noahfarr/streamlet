<h1 align="center">
  <b>Strelax</b><br>
  <b>Streaming Reinforcement Learning in JAX</b><br>
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" /></a>
  <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/powered%20by-JAX-9cf.svg" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>

Most deep RL is built around large replay buffers and big batched updates. `Strelax` takes the opposite approach. It implements *streaming* RL, where the agent learns online from each transition the moment it arrives, batch size one, no replay buffer, using eligibility traces and optimizers designed to stay stable in that regime. Everything is written in JAX, so every algorithm is `jit`-compatible and `vmap`s cleanly over random seeds, turning a streaming agent that learns from one transition at a time into a fast, fully reproducible experiment.

<h2> ✨ Features </h2>

| | Details |
|---|---|
| 🤖 **Algorithms** | [Stream Q(λ)](https://arxiv.org/abs/2410.14606), [Stream AC(λ)](https://arxiv.org/abs/2410.14606), Stream SARSA(λ), [QRC](https://arxiv.org/abs/2007.00611) (Q-learning with regularized corrections), and [AVG](https://arxiv.org/abs/2411.15370) (action-value gradient for continuous control) — all online, with eligibility traces and no replay buffer |
| ⚙️ **Optimizers** | [ObGD](https://arxiv.org/abs/2410.14606) (overshooting-bounded gradient descent), [`AdaptiveQ`](https://arxiv.org/abs/2605.06764) (Adam-style adaptive step sizes), `IntentionalOptimizer`, and an [`optax`](https://github.com/google-deepmind/optax) wrapper for standard optimizers |
| 🧬 **Networks** | Flax heads for discrete and continuous Q-values, value functions, and categorical / Gaussian policies (state-dependent, state-independent, and squashed). [Sparse initialization](https://arxiv.org/abs/2410.14606). Composable `torso` → `head` pipeline |
| 🎮 **Environments** | [Gymnax](https://github.com/RobertTLange/gymnax), [Brax](https://github.com/google/brax), [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment), and [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) behind a single `make("namespace::env_id")` entry point |
| 🧰 **Wrappers** | Observation / reward normalization, episode-statistics recording, sticky actions |
| 📊 **Logging** | In-graph structured logging via [`lox`](https://github.com/huterguier/lox) |

<h2> 📥 Installation</h2>

`Strelax` uses [`uv`](https://github.com/astral-sh/uv) and requires Python ≥ 3.12. Clone and sync:

```bash
git clone https://github.com/noahfarr/strelax.git
cd strelax
uv sync
```

This installs JAX with CUDA 12 support on Linux and CPU/Metal JAX on macOS. To add `Strelax` to an existing project:

```bash
uv add git+https://github.com/noahfarr/strelax.git
```

<h2> 🚀 Quick Start</h2>

Train a streaming Q(λ) agent on MinAtar Breakout:

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
from strelax.algorithms import StreamQ, StreamQConfig
from strelax.environments import environment
from strelax.networks import Flatten, heads, sparse
from strelax.optimizers import OBGD, OBGDConfig

env, env_params = environment.make("gymnax::Breakout-MinAtar")
num_actions = env.action_space(env_params).n

cfg = StreamQConfig(num_envs=1, gamma=0.99, trace_lambda=0.8)

sparse_init = sparse(sparsity=0.9)
q_network = nn.Sequential([
    nn.Conv(16, (3, 3), padding="VALID", kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
    Flatten(start_dim=-3),
    nn.Dense(128, kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
    heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=sparse_init),
])

optimizer = OBGD(OBGDConfig(lr=1.0, kappa=2.0))
epsilon = lambda step: jnp.maximum(1.0 - step / 1e6, 0.01)

agent = StreamQ(cfg, env, env_params, q_network, epsilon, optimizer)
key = jax.random.key(0)
key, init_key = jax.random.split(key)
state = agent.init(init_key)
key, train_key = jax.random.split(key)
state = agent.train(train_key, state, num_steps=100_000)
```

Every algorithm exposes the same interface: `init` → `warmup` (optional) → `train` → `evaluate`. See `examples/` for complete scripts with logging and evaluation.

<h2> 💡 Advanced Usage</h2>

`Strelax` is built to scale experiments across seeds. Because every method is a pure function, you can `vmap` over seeds and use [`lox`](https://github.com/huterguier/lox) to collect metrics logged from *inside* the training scan — no Python-side bookkeeping:

```python
import jax
import lox

# agent built as above
num_seeds = 8
init = jax.vmap(agent.init)
train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

key = jax.random.key(0)
key, init_key = jax.random.split(key)
state = init(jax.random.split(init_key, num_seeds))

state, logs = train(jax.random.split(key, num_seeds), state, 100_000)
# logs is a pytree of per-step metrics (TD error, value, trace norm, ...), batched over seeds
```

Swapping in `StreamAC` with the `IntentionalOptimizer` on a Brax control task is a matter of changing the imports — see `examples/intentional_ac_brax.py`.

<h2> 📂 Project Structure</h2>

```
strelax/
├─ examples/          # Runnable scripts (Stream Q / SARSA / AC, QRC, AVG on MinAtar & Brax)
├─ strelax/
   ├─ algorithms/     # StreamQ, StreamSARSA, StreamAC, QRC, AVG
   ├─ optimizers/     # ObGD, AdaptiveQ, IntentionalOptimizer, optax wrapper
   ├─ environments/   # Gymnax / Brax / ALE / Gymnasium adapters + wrappers
   ├─ networks/       # heads, layers, initializers
   └─ utils/          # Timestep, Transition, TD-error scaler, helpers
```

<h2> 📄 License</h2>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<h2> 📚 Citation</h2>

If you use `Strelax` for your work, please cite:
```
@software{strelax2026github,
  title   = {Strelax: Streaming Reinforcement Learning in JAX},
  author  = {Noah Farr},
  year    = {2026},
  url     = {https://github.com/noahfarr/strelax}
}
```

<h2> 🙏 Acknowledgments</h2>

Streaming algorithms, ObGD, and sparse initialization follow [Elsayed et al., *Streaming Deep Reinforcement Learning Finally Works*](https://arxiv.org/abs/2410.14606). Special thanks to [@huterguier](https://github.com/huterguier) for [`lox`](https://github.com/huterguier/lox) and for discussions on the API design.
