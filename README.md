<h1 align="center">
  <b>Streamlet</b><br>
  <b>Streaming Reinforcement Learning in JAX</b><br>
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" /></a>
  <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/powered%20by-JAX-9cf.svg" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>

Most deep RL is built around large replay buffers and big batched updates. `Streamlet` takes the opposite approach. It implements *streaming* RL, where the agent learns online from each transition the moment it arrives, batch size one, no replay buffer, using eligibility traces and optimizers designed to stay stable in that regime. Everything is written in JAX, so every algorithm is `jit`-compatible and `vmap`s cleanly over random seeds, turning a streaming agent that learns from one transition at a time into a fast, fully reproducible experiment.

<h2> ✨ Features </h2>

| | Details |
|---|---|
| 🤖 **Algorithms** | [Q(λ)](https://arxiv.org/abs/2410.14606), [AC(λ)](https://arxiv.org/abs/2410.14606), [SARSA(λ)](https://arxiv.org/abs/2410.14606), [TD(λ)](https://arxiv.org/abs/2410.14606), [QRC(λ)](https://arxiv.org/abs/2507.09087), [AVG(λ)](https://arxiv.org/abs/2411.15370), soft (entropy-regularized) Q(λ)/AC(λ), and recurrent Q(λ)/AC(λ)/QRC(λ)/TD(λ) for POMDPs — all online, with eligibility traces and no replay buffer |
| ⚙️ **Optimizers** | [ObGD](https://arxiv.org/abs/2410.14606), [`Adaptive`](https://arxiv.org/abs/2605.06764), [`Implicit`](https://arxiv.org/abs/2505.01361), [`Intentional`](https://arxiv.org/abs/2604.19033), `Calibrated`, `AlphaBound`, and an [`optax`](https://github.com/google-deepmind/optax) wrapper for standard optimizers |
| 🎮 **Environments** | [Gymnax](https://github.com/RobertTLange/gymnax), [Brax](https://github.com/google/brax), [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), [Craftax](https://github.com/MichaelTMatthews/Craftax), [PufferLib](https://github.com/PufferAI/PufferLib), a native MinAtar reimplementation, an Animax prediction-benchmark namespace, and the built-in [ETT](https://github.com/zhouhaoyi/ETDataset) time-series prediction dataset — all behind a single `make("namespace::env_id")` entry point |
| 🧰 **Wrappers** | Observation / reward normalization, observation traces, episode-statistics recording, sticky actions |
| 📊 **Logging** | In-graph structured logging via [`lox`](https://github.com/huterguier/lox), plus `streamlet.loggers` for a live terminal dashboard, Weights & Biases, and checkpointing |

<h2> 📥 Installation</h2>

`Streamlet` uses [`uv`](https://github.com/astral-sh/uv) and requires Python ≥ 3.12. Clone and sync:

```bash
git clone https://github.com/noahfarr/streamlet.git
cd streamlet
uv sync
```

This installs JAX with CUDA 12 support on Linux and CPU/Metal JAX on macOS. To add `Streamlet` to an existing project:

```bash
uv add git+https://github.com/noahfarr/streamlet.git
```

<h2> 🚀 Quick Start</h2>

Train a streaming Q(λ) agent on MinAtar Breakout:

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
from streamlet.algorithms import QLambda, QLambdaConfig
from streamlet.environments import environment
from streamlet.networks import Flatten, sparse
from streamlet.optimizers import ObGD, ObGDConfig

env, env_params = environment.make("gymnax::Breakout-MinAtar")
num_actions = env.action_space(env_params).n

cfg = QLambdaConfig(gamma=0.99, trace_lambda=0.8)

sparse_init = sparse(sparsity=0.9)
q_network = nn.Sequential([
    nn.Conv(16, (3, 3), padding="VALID", kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
    Flatten(start_dim=-3),
    nn.Dense(128, kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
    nn.Dense(num_actions, kernel_init=sparse_init),
])

optimizer = ObGD(ObGDConfig(lr=1.0, kappa=2.0))
epsilon = lambda step: jnp.maximum(1.0 - step / 1e6, 0.01)

agent = QLambda(cfg, env, env_params, q_network, epsilon, optimizer)
key = jax.random.key(0)
key, init_key = jax.random.split(key)
state = agent.init(init_key)
key, train_key = jax.random.split(key)
state = agent.train(train_key, state, num_steps=100_000)
```

Every algorithm exposes the same interface: `init` → `train` → `evaluate`, all `jit`/`vmap`-compatible so you can parallelize across seeds with `jax.vmap`. See `examples/` for complete scripts with logging and evaluation.

<h2> 📂 Project Structure</h2>

```
streamlet/
├─ examples/          # Runnable scripts (Q / SARSA / AC / TD, QRC, AVG on MinAtar, Brax & ETT)
├─ streamlet/
   ├─ algorithms/     # Q/SARSA/AC/TD/QRC/AVGLambda, soft & recurrent variants
   ├─ optimizers/     # ObGD, Adaptive, Implicit, Intentional, Calibrated, AlphaBound, optax wrapper
   ├─ environments/   # Gymnax / Brax / ALE / Gymnasium / Craftax / PufferLib / Animax / native MinAtar / ETT adapters + wrappers
   ├─ networks/       # layers, initializers
   ├─ loggers/        # Dashboard, Weights & Biases, checkpoint, and multi-logger
   └─ utils/          # Timestep, Transition, TD-error scaler, helpers
```

<h2> 📄 License</h2>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<h2> 📚 Citation</h2>

If you use `Streamlet` for your work, please cite:
```
@software{streamlet2026github,
  title   = {Streamlet: Streaming Reinforcement Learning in JAX},
  author  = {Noah Farr},
  year    = {2026},
  url     = {https://github.com/noahfarr/streamlet}
}
```

<h2> 🙏 Acknowledgments</h2>

Streaming algorithms, ObGD, and sparse initialization follow [Elsayed et al., *Streaming Deep Reinforcement Learning Finally Works*](https://arxiv.org/abs/2410.14606).
