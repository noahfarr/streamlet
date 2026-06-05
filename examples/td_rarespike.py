"""Rare-spike linear TD(0): mean-square (in)stability and the coefficient of variation.

Scalar feature phi in {M w.p. p, 1 w.p. 1-p}, r == 0 (so w* = 0, the iterate IS the error),
lambda == 0. The per-sample interaction is X = phi (phi - gamma phi'), and the per-step
second-moment multiplier is rho(alpha) = E[(1 - alpha X)^2] = 1 - 2 alpha A + alpha^2 E[X^2].

Three step sizes, three fates:

  - Mean step  alpha = 1/A           -> rho = CV^2(X).   DIVERGES once std(X) > mean(X).
  - Calibrated alpha* = A/E[X^2]      -> rho = CV^2/(1+CV^2) < 1 for any CV^2 (variance-optimal).
  - AlphaBound (Dabney & Barto 2012)  -> running minimum of 1/|X_t|, init 1.

The mean step is the idealized "expectation-based" baseline: it steps at the inverse of the
*mean* interaction, so its second moment grows the moment CV^2 > 1. The literal AlphaBound
optimizer instead caps the step at the inverse of each *instantaneous* interaction, so a
single spike drives alpha down to ~1/max|X| and it stays mean-square stable -- but it
overshrinks (alpha well below the variance-optimal A/E[X^2]). Calibrated hits the largest
mean-square-stable step without per-sample over-conservatism.

This runs the literal AlphaBound and Calibrated through the real TDLambda(lambda=0) stack on
`rarespike::Spike`, measures their online step sizes, and overlays them on the closed-form
mean-square multiplier rho.

Note on measurement: with gamma = 0 the *typical* trajectory of the mean step contracts
(non-spike factor (1-1/A)^2 ~ 0.04) while E[w^2] diverges only through extremely rare
near-all-spike paths. A naive Monte-Carlo average of compounded w_t^2 is therefore tail
dominated and understates the divergence -- which is exactly why an expectation-based step
size is dangerous: typical runs look fine while the second moment explodes. We plot the
closed-form rho^t and validate the optimizers' online step sizes instead.
"""

import argparse
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
import matplotlib.pyplot as plt
import numpy as np

from rarespike_analysis import (
    mean_interaction,
    rho,
    second_moment_interaction,
    stability,
)

from streax.algorithms import TDLambda, TDLambdaConfig
from streax.environments import environment
from streax.optimizers import AlphaBound, AlphaBoundConfig, Calibrated, CalibratedConfig

parser = argparse.ArgumentParser()
parser.add_argument("--M", type=float, default=3.0, help="Rare spike magnitude.")
parser.add_argument("--p", type=float, default=0.03, help="Spike probability.")
parser.add_argument(
    "--gamma",
    type=float,
    default=0.0,
    help="Discount (0.0 is cleanest; 0.9 pushes CV^2 ~ 20).",
)
parser.add_argument("--num-seeds", type=int, default=2048)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

M, p, gamma = args.M, args.p, args.gamma
num_seeds, steps = args.num_seeds, args.steps

theory = stability(M, p, gamma)
A = mean_interaction(M, p, gamma)
E_X2 = second_moment_interaction(M, p, gamma)

env, env_params = environment.make("rarespike::Spike", M=M, p=p)
config = TDLambdaConfig(gamma=gamma, trace_lambda=0.0)


def value_network():
    # Linear scalar value v = w . phi with w_0 = 1, no bias.
    return nn.Dense(1, use_bias=False, kernel_init=nn.initializers.constant(1.0))


def seed_calibrated(opt_state):
    # Seed first/second moments at (A, E[X^2]) so alpha = A/E[X^2] from step 1.
    return opt_state.replace(
        m_hat=jnp.full_like(opt_state.m_hat, A),
        s_hat=jnp.full_like(opt_state.s_hat, E_X2),
        y_hat=jnp.zeros_like(opt_state.y_hat),
        step=jnp.full_like(opt_state.step, 1e6),
    )


def run(optimizer, seed_fn=None):
    agent = TDLambda(config, env, env_params, value_network(), optimizer)
    init = jax.vmap(agent.init)
    train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))

    key = jax.random.key(args.seed)
    key, init_key = jax.random.split(key)
    state = init(jax.random.split(init_key, num_seeds))
    if seed_fn is not None:
        state = state.replace(value_optimizer_state=seed_fn(state.value_optimizer_state))

    key, train_key = jax.random.split(key)
    state, logs = train(jax.random.split(train_key, num_seeds), state, steps)
    jax.block_until_ready(state)
    return logs


def step_sizes(logs, name):
    return np.asarray(logs[f"{name}/step_size"]).reshape(num_seeds, steps).mean(axis=0)


ab_logs = run(AlphaBound(cfg=AlphaBoundConfig()))
cal_logs = run(Calibrated(cfg=CalibratedConfig(nu=0.0, alpha_max=1.0)), seed_calibrated)

alpha_ab = step_sizes(ab_logs, "alpha_bound")
alpha_cal = step_sizes(cal_logs, "calibrated")

# Literal AlphaBound is a running minimum of 1/|X_t|; it converges to 1/max|X|
# (= 1/M^2 for gamma=0), the over-conservative steady cap.
alpha_ab_now = float(alpha_ab[-1])
alpha_ab_cap = 1.0 / max(M * (M - gamma), M, 1.0)
rho_ab_cap = rho(alpha_ab_cap, M, p, gamma)

print(f"rare-spike linear TD(0):  M={M}  p={p}  gamma={gamma}")
print(f"  A = {A:.4f}   E[X^2] = {E_X2:.4f}   CV^2 = {theory.cv2:.4f}")
print("  step size alpha            ->  mean-square multiplier rho:")
print(
    f"    mean step   1/A         = {theory.alpha_mean_step:.4f}  ->  "
    f"rho = CV^2 = {theory.rho_mean_step:.4f}  "
    f"({'DIVERGES' if theory.rho_mean_step > 1 else 'contracts'})"
)
print(
    f"    Calibrated  A/E[X^2]    = {alpha_cal.mean():.4f}  ->  "
    f"rho = CV^2/(1+CV^2) = {theory.rho_calibrated:.4f}  "
    f"(contracts; variance-optimal, the rho-minimizing step)"
)
print(
    f"    AlphaBound (measured)   = {alpha_ab_now:.4f}  (step {steps}, still converging)"
)
print(
    f"    AlphaBound  1/max|X|    = {alpha_ab_cap:.4f}  ->  "
    f"rho = {rho_ab_cap:.4f}  "
    f"(MS-stable steady cap; over-conservative vs Calibrated)"
)

t = np.arange(steps + 1)
ew2_mean = theory.rho_mean_step**t
ew2_cal = theory.rho_calibrated**t

plot_dir = Path("plots") / "rarespike"
plot_dir.mkdir(parents=True, exist_ok=True)

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))

ax0.set_yscale("log")
ax0.axhline(1.0, color="gray", linestyle=":", linewidth=1.5)
ax0.plot(
    t, ew2_mean, color="tab:red", linewidth=3.0,
    label=rf"mean step $1/A$:  $\rho=CV^2={theory.rho_mean_step:.2f}$",
)
ax0.plot(
    t, ew2_cal, color="tab:blue", linewidth=3.0,
    label=rf"Calibrated:  $\rho=\frac{{CV^2}}{{1+CV^2}}={theory.rho_calibrated:.2f}$",
)
ax0.set_xlabel("Time Step", fontsize=16)
ax0.set_ylabel(r"$\mathbb{E}[w_t^2]$ (closed form $\rho^t$)", fontsize=16)
ax0.set_title(
    rf"Mean-square: $M={M},\ p={p},\ \gamma={gamma},\ CV^2={theory.cv2:.2f}$",
    fontsize=15,
)
ax0.legend(fontsize=12)

ax1.axhline(
    theory.alpha_mean_step, color="tab:red", linestyle="--", linewidth=2.0,
    label=r"mean step $1/A$ (diverges)",
)
ax1.axhline(
    theory.alpha_calibrated, color="tab:blue", linestyle="--", linewidth=2.0,
    label=r"Calibrated $A/\mathbb{E}[X^2]$",
)
ax1.plot(
    np.arange(steps), alpha_ab, color="tab:green", linewidth=2.5,
    label="AlphaBound (measured)",
)
ax1.plot(np.arange(steps), alpha_cal, color="tab:blue", linewidth=2.5, alpha=0.8)
ax1.set_xlabel("Time Step", fontsize=16)
ax1.set_ylabel(r"online step size $\alpha$", fontsize=16)
ax1.set_title("Online step size: AlphaBound caps below the variance-optimal", fontsize=14)
ax1.legend(fontsize=12)

fig.tight_layout()
fig.savefig(plot_dir / "keystone.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved plot to {plot_dir / 'keystone.png'}")
