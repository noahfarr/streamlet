"""Per-seed investigation of the Asterix-MinAtar / QLambda divergence.

Runs a focused diagnostic (default 200k steps) comparing, on identical env
seeds and an identical network/normalization stack (sticky actions matched):

  - obgd                : ObGD baseline (known to learn Asterix).
  - calibrated            : Calibrated with preconditioning ON  (beta=0.999).
  - calibrated-no-precond : Calibrated with preconditioning OFF (beta=1.0 freezes
                          v at preconditioning_init, so P is a constant ~= I;
                          because alpha self-normalises the uniform scale this
                          is exactly the pre-preconditioning Calibrated).
  - calibrated-alphacap   : Calibrated (precond off) with alpha_max=1e-2, to test
                          whether an early oversized step triggers divergence.

For each config it reports, PER SEED (aggregate means are poisoned by a single
diverging seed), the divergence onset step and a finite-guarded summary, then
saves per-seed diagnostic plots.
"""

import argparse
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import matplotlib.pyplot as plt
import numpy as np

from streamlet.algorithms import QLambda, QLambdaConfig
from streamlet.environments import environment
from streamlet.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from streamlet.networks import Flatten, sparse
from streamlet.optimizers import Calibrated, CalibratedConfig, ObGD, ObGDConfig

parser = argparse.ArgumentParser()
parser.add_argument("--env-id", default="gymnax::Asterix-MinAtar")
parser.add_argument("--steps", type=int, default=200_000)
parser.add_argument("--num-seeds", type=int, default=5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--diverge-threshold",
    type=float,
    default=1e3,
    help="|q_value| above this (or non-finite) counts as divergence onset.",
)
args = parser.parse_args()

# Epsilon schedule is deliberately tied to the production 5M-step horizon so the
# exploration regime (epsilon ~0.9 around ~100k steps) matches the real run even
# though we only roll out `--steps`.
TOTAL_TIMESTEPS = 5_000_000
gamma = 0.99
trace_lambda = 0.8

epsilon_start, epsilon_end, exploration_fraction = 1.0, 0.01, 0.2
exploration_steps = exploration_fraction * TOTAL_TIMESTEPS


def epsilon_schedule(step):
    frac = jnp.minimum(step / exploration_steps, 1.0)
    return epsilon_start + frac * (epsilon_end - epsilon_start)


def make_env():
    env, env_params = environment.make(args.env_id)
    env = StickyActionWrapper(env)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=gamma)
    return env, env_params


def make_network(num_actions):
    sparse_init = sparse(sparsity=0.9)
    return nn.Sequential(
        [
            nn.Conv(
                16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init
            ),
            nn.LayerNorm(),
            nn.leaky_relu,
            Flatten(start_dim=-3),
            nn.Dense(128, kernel_init=sparse_init),
            nn.LayerNorm(),
            nn.leaky_relu,
            nn.Dense(num_actions, kernel_init=sparse_init),
        ]
    )


CONFIGS = {
    "obgd": lambda: ObGD(
        cfg=ObGDConfig(lr=1.0, kappa=2.0, beta2=0.999, eps=1e-8, adaptive=False)
    ),
    "calibrated": lambda: Calibrated(cfg=CalibratedConfig(beta=0.999)),
    "calibrated-no-precond": lambda: Calibrated(
        cfg=CalibratedConfig(beta=1.0)
    ),
    "calibrated-alphacap": lambda: Calibrated(
        cfg=CalibratedConfig(beta=1.0, alpha_max=1e-2)
    ),
}


def run_config(name, optimizer, env, env_params, num_actions, init_key, train_key):
    """Train `--num_seeds` seeds and return host-side numpy logs (seeds x steps)."""
    config = QLambdaConfig(gamma=gamma, trace_lambda=trace_lambda)
    network = make_network(num_actions)
    agent = QLambda(config, env, env_params, network, epsilon_schedule, optimizer)

    init = jax.vmap(agent.init)
    train = jax.jit(jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None)), static_argnums=2, donate_argnums=1)

    # SAME init/train keys across configs -> identical network init and env-seed
    # stream; only the learning rule differs.
    state = init(jax.random.split(init_key, args.num_seeds))
    state = jax.tree.map(lambda x: jax.lax.convert_element_type(x, x.dtype), state)
    state, logs = train(jax.random.split(train_key, args.num_seeds), state, args.steps)
    jax.block_until_ready((state, logs))

    return {k: np.asarray(v) for k, v in logs.items()}


def onset_step(q_per_step, threshold):
    """First step index where |q| exceeds threshold or is non-finite; -1 if never."""
    bad = ~np.isfinite(q_per_step) | (np.abs(q_per_step) > threshold)
    idx = np.argmax(bad)
    return int(idx) if bad[idx] else -1


def finite_summary(arr):
    """Mean/median over finite entries only (NaN-guarded)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(finite)), float(np.median(finite))


def report(name, logs, threshold):
    q = logs["q_network/q_value"]  # (seeds, steps)
    num_seeds = q.shape[0]
    print(f"\n===== {name} =====")
    onsets = []
    for s in range(num_seeds):
        onset = onset_step(q[s], threshold)
        onsets.append(onset)
        q_mean, q_med = finite_summary(q[s])
        tag = f"diverged @ step {onset:>7d}" if onset >= 0 else "stable          "
        print(
            f"  seed {s}: {tag} | q finite mean={q_mean:11.3f} median={q_med:9.3f} "
            f"| finite frac={np.isfinite(q[s]).mean():.4f}"
        )
    diverged = [o for o in onsets if o >= 0]
    frac_div = len(diverged) / num_seeds
    print(
        f"  -> fraction diverged: {frac_div:.2f} ({len(diverged)}/{num_seeds})"
        + (f", earliest onset step {min(diverged)}" if diverged else "")
    )
    return onsets


def plot_config(name, logs, onsets, plot_dir):
    keys = [
        ("q_network/q_value", "q_value"),
        ("q_network/td_error", "td_error"),
        ("calibrated/m_hat", "m_hat"),
        ("calibrated/s_hat", "s_hat"),
        ("calibrated/step_size", "step_size"),
        ("calibrated/update_norm", "update_norm"),
        ("q_trace/trace_norm", "trace_norm"),
        ("obgd/step_size", "step_size"),
        ("obgd/update_norm", "update_norm"),
    ]
    present = [(k, lbl) for k, lbl in keys if k in logs]
    n = len(present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    num_seeds = logs["q_network/q_value"].shape[0]
    for ax, (k, lbl) in zip(axes, present):
        arr = logs[k]
        for s in range(num_seeds):
            y = np.where(np.isfinite(arr[s]), arr[s], np.nan)
            ax.plot(y, lw=0.7, alpha=0.8, label=f"seed {s}")
        ax.set_title(f"{name} | {k}")
        ax.set_xlabel("step")
        # symlog handles the huge magnitudes near divergence without clipping.
        ax.set_yscale("symlog")
        if k == "q_network/q_value":
            ax.legend(fontsize=6, ncol=2)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    out = plot_dir / f"{name}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved plot -> {out}")


def main():
    env, env_params = make_env()
    num_actions = env.action_space(env_params).n

    key = jax.random.key(args.seed)
    init_key, train_key = jax.random.split(key)

    plot_dir = Path("plots") / "investigation" / args.env_id.replace("::", "_")
    plot_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"env={args.env_id} steps={args.steps} seeds={args.num_seeds} "
        f"threshold={args.diverge_threshold}"
    )
    summary = {}
    for name, build in CONFIGS.items():
        logs = run_config(
            name, build(), env, env_params, num_actions, init_key, train_key
        )
        onsets = report(name, logs, args.diverge_threshold)
        plot_config(name, logs, onsets, plot_dir)
        summary[name] = onsets
        del logs  # free host memory before the next config

    print("\n===== SUMMARY (onset step per seed; -1 = stable) =====")
    for name, onsets in summary.items():
        print(f"  {name:22s}: {onsets}")


if __name__ == "__main__":
    main()
