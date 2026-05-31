<h1 align="center">
  <b>Stremax</b><br>
  <b>Streaming Reinforcement Learning in JAX</b><br>
</h1>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12-blue.svg" /></a>
  <a href="https://github.com/jax-ml/jax"><img src="https://img.shields.io/badge/powered%20by-JAX-9cf.svg" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-blue.svg" /></a>
</p>

Most deep RL is built around large replay buffers and big batched updates. `Stremax` takes the opposite approach. It implements *streaming* RL, where the agent learns online from each transition the moment it arrives, batch size one, no replay buffer, using eligibility traces and optimizers designed to stay stable in that regime. Everything is written in JAX, so every algorithm is `jit`-compatible and `vmap`s cleanly over random seeds, turning a streaming agent that learns from one transition at a time into a fast, fully reproducible experiment.

<h2> ✨ Features </h2>

| | Details |
|---|---|
| 🤖 **Algorithms** | [Stream Q(λ)](https://arxiv.org/abs/2410.14606), [Stream AC(λ)](https://arxiv.org/abs/2410.14606), [Stream SARSA(λ)](https://arxiv.org/abs/2410.14606), [Stream TD(λ)](https://arxiv.org/abs/2410.14606), [QRC](https://arxiv.org/abs/2507.09087), and [AVG](https://arxiv.org/abs/2411.15370) — all online, with eligibility traces and no replay buffer |
| ⚙️ **Optimizers** | [ObGD](https://arxiv.org/abs/2410.14606), [`AdaptiveQ`](https://arxiv.org/abs/2605.06764), [`Implicit`](https://arxiv.org/abs/2505.01361), [`Intentional`](https://arxiv.org/abs/2604.19033), and an [`optax`](https://github.com/google-deepmind/optax) wrapper for standard optimizers |
| 🎮 **Environments** | [Gymnax](https://github.com/RobertTLange/gymnax), [Brax](https://github.com/google/brax), [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment), [Gymnasium](https://github.com/Farama-Foundation/Gymnasium), and the built-in [ETT](https://github.com/zhouhaoyi/ETDataset) time-series prediction dataset behind a single `make("namespace::env_id")` entry point |
| 🧰 **Wrappers** | Observation / reward normalization, observation traces, episode-statistics recording, sticky actions |
| 📊 **Logging** | In-graph structured logging via [`lox`](https://github.com/huterguier/lox) |

<h2> 📥 Installation</h2>

`Stremax` uses [`uv`](https://github.com/astral-sh/uv) and requires Python ≥ 3.12. Clone and sync:

```bash
git clone https://github.com/noahfarr/stremax.git
cd stremax
uv sync
```

This installs JAX with CUDA 12 support on Linux and CPU/Metal JAX on macOS. To add `Stremax` to an existing project:

```bash
uv add git+https://github.com/noahfarr/stremax.git
```

<h2> 🚀 Quick Start</h2>

Train a streaming Q(λ) agent on MinAtar Breakout:

```python
import flax.linen as nn
import jax
import jax.numpy as jnp
from stremax.algorithms import QLambda, QLambdaConfig
from stremax.environments import environment
from stremax.networks import Flatten, heads, sparse
from stremax.optimizers import ObGD, ObGDConfig

env, env_params = environment.make("gymnax::Breakout-MinAtar")
num_actions = env.action_space(env_params).n

cfg = QLambdaConfig(num_envs=1, gamma=0.99, trace_lambda=0.8)

sparse_init = sparse(sparsity=0.9)
q_network = nn.Sequential([
    nn.Conv(16, (3, 3), padding="VALID", kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
    Flatten(start_dim=-3),
    nn.Dense(128, kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
    heads.DiscreteQNetwork(action_dim=num_actions, kernel_init=sparse_init),
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

Every algorithm exposes the same interface: `init` → `warmup` (optional) → `train` → `evaluate`. See `examples/` for complete scripts with logging and evaluation.

<h2> 🖥️ Launching on a cluster</h2>

`scripts/launch.py` submits an example script to SLURM as an array job (one task per env-id, and per `--eta` for examples that take it) via [`slurmpilot`](https://github.com/geoalgo/slurmpilot). You run it from your laptop; slurmpilot ships the repo to the cluster over SSH and calls `sbatch` there. The upload is filtered by this repo's `.gitignore`, so `.venv/` and `wandb/` are not copied.

Install the launch tooling (pulled from the fork that adds `.gitignore`-aware uploads), then point slurmpilot at your cluster:

```bash
uv sync --extra cluster

mkdir -p ~/slurmpilot/config/clusters
cp scripts/slurmpilot/general.yaml      ~/slurmpilot/config/general.yaml
cp scripts/slurmpilot/clusters/ias.yaml ~/slurmpilot/config/clusters/ias.yaml
# edit ~/slurmpilot/config/clusters/ias.yaml and set `host:` (and `user:`)
```

Then launch (use `--dry-run` first to print the sweep and inspect the generated `slurm_script.sh` without submitting):

```bash
# Measured eta sweep across the 4 MinAtar envs on CPU, with wandb (the old launch.sh default)
uv run python scripts/launch.py

# A different example on GPU, no eta sweep, no wandb
uv run python scripts/launch.py --example stream_q_minatar --device gpu --no-wandb

# MinAtar Q-learning across ALL optimizers (ObGD, Measured, Implicit, Intentional, AdaptiveQ),
# one array job per optimizer
uv run python scripts/minatar.py
```

Each node runs `uv sync --frozen` against the shipped `uv.lock` before training. If your compute nodes have no internet, override the setup step to use a pre-built environment, e.g. `--setup 'source ~/stremax-env/bin/activate'`.

<h2> 📂 Project Structure</h2>

```
stremax/
├─ examples/          # Runnable scripts (Stream Q / SARSA / AC / TD, QRC, AVG on MinAtar, Brax & ETT)
├─ scripts/           # slurmpilot launchers (launch.py, minatar.py) + cluster config templates
├─ stremax/
   ├─ algorithms/     # QLambda, SARSALambda, ACLambda, TDLambda, QRC, AVG
   ├─ optimizers/     # ObGD, AdaptiveQ, Implicit, Intentional, optax wrapper
   ├─ environments/   # Gymnax / Brax / ALE / Gymnasium / ETT adapters + wrappers
   ├─ networks/       # heads, layers, initializers
   └─ utils/          # Timestep, Transition, TD-error scaler, helpers
```

<h2> 📄 License</h2>

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<h2> 📚 Citation</h2>

If you use `Stremax` for your work, please cite:
```
@software{stremax2026github,
  title   = {Stremax: Streaming Reinforcement Learning in JAX},
  author  = {Noah Farr},
  year    = {2026},
  url     = {https://github.com/noahfarr/stremax}
}
```

<h2> 🙏 Acknowledgments</h2>

Streaming algorithms, ObGD, and sparse initialization follow [Elsayed et al., *Streaming Deep Reinforcement Learning Finally Works*](https://arxiv.org/abs/2410.14606).
