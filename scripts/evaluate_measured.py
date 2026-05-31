import argparse
import time

import flax.linen as nn
import jax
import jax.numpy as jnp
import lox
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
from stremax.optimizers import (
    Implicit,
    ImplicitConfig,
    Intentional,
    IntentionalConfig,
    Measured,
    MeasuredConfig,
    ObGD,
    ObGDConfig,
)

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=68_000)
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--eta", type=float, default=0.5)
p.add_argument("--nu", type=float, default=0.01)
p.add_argument("--beta", type=float, default=0.999)
p.add_argument("--compare", action="store_true", help="Also run the baseline optimizers.")
p.add_argument("--sweep", action="store_true", help="Sweep measured hyperparameters.")
args = p.parse_args()

env_id = "ett::ETTm2"
gamma = 0.99
trace_lambda = 0.8


def build_env():
    env, env_params = environment.make(env_id, dataset_path="ETTm2.csv")
    env = RecordEpisodeStatistics(env, gamma=gamma)
    env = ObservationTracesWrapper(env, beta=args.beta)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env, gamma=gamma)
    return env, env_params


def value_network():
    sparse_init = sparse(sparsity=0.9)
    return nn.Sequential([
        nn.Dense(128, kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
        nn.Dense(128, kernel_init=sparse_init), nn.LayerNorm(), nn.leaky_relu,
        heads.VNetwork(kernel_init=sparse_init),
    ])


S, T = args.seeds, args.steps
config = StreamTDConfig(num_envs=1, gamma=gamma, trace_lambda=trace_lambda)


def run(optimizer):
    env, env_params = build_env()
    agent = StreamTD(config, env, env_params, value_network(), optimizer)
    init = jax.vmap(agent.init)
    train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))
    key = jax.random.key(0)
    key, init_key = jax.random.split(key)
    state = init(jax.random.split(init_key, S))
    key, train_key = jax.random.split(key)
    t0 = time.perf_counter()
    state, logs = train(jax.random.split(train_key, S), state, T)
    jax.block_until_ready(state)
    return logs, time.perf_counter() - t0


def metrics(logs):
    std = np.asarray(logs["normalize_reward/std"]).reshape(S, T)
    pred = np.asarray(logs["value/value"]).reshape(S, T) * std
    cumulants = np.asarray(logs["value/cumulant"]).reshape(S, T) * std
    td = np.abs(np.asarray(logs["value/td_error"]).reshape(S, T))
    actual = np.zeros((S, T))
    ret = np.zeros(S)
    for t in reversed(range(T)):
        ret = ret * gamma + cumulants[:, t]
        actual[:, t] = ret
    n = T // 10
    err = pred - actual
    rmse_first = float(np.sqrt(np.mean(err[:, :n] ** 2)))
    rmse_last = float(np.sqrt(np.mean(err[:, -n:] ** 2)))
    return rmse_first, rmse_last, td[:, :n].mean(), td[:, -n:].mean(), np.isfinite(pred).all()


def alpha_early_late(logs):
    ss = np.asarray(logs["measured/step_size"]).reshape(S, T)
    n = T // 10
    return ss[:, :n].mean(), ss[:, -n:].mean()


if args.sweep:
    base = dict(eta=0.5, nu=0.01, beta=0.999)
    grid = [
        base,
        {**base, "beta": 0.99}, {**base, "beta": 0.95}, {**base, "beta": 0.9},
        {**base, "eta": 0.25}, {**base, "eta": 1.0},
        {**base, "nu": 0.001}, {**base, "nu": 0.1},
        {"eta": 1.0, "nu": 0.01, "beta": 0.95},
    ]
    print(f"\n=== {env_id}  measured sweep  |  seeds={S}  steps={T:_} ===\n")
    hdr = f"{'eta':>5}{'nu':>7}{'beta':>7}{'RMSE first':>12}{'RMSE last':>11}{'|TD|first':>11}{'|TD|last':>10}{'a_early':>10}{'a_late':>10}"
    print(hdr)
    print("-" * len(hdr))
    for cfg in grid:
        logs, _ = run(Measured(cfg=MeasuredConfig(**cfg)))
        rf, rl, tf, tl, fin = metrics(logs)
        ae, al = alpha_early_late(logs)
        flag = "" if fin else "  NONFINITE"
        print(f"{cfg['eta']:>5}{cfg['nu']:>7}{cfg['beta']:>7}{rf:>12.2f}{rl:>11.2f}{tf:>11.4f}{tl:>10.4f}{ae:>10.2e}{al:>10.2e}{flag}")
    raise SystemExit

measured = Measured(cfg=MeasuredConfig(eta=args.eta, nu=args.nu, beta=args.beta))
optimizers = {f"measured (eta={args.eta},nu={args.nu})": measured}
if args.compare:
    optimizers.update({
        "implicit (lr=1.0)": Implicit(cfg=ImplicitConfig(lr=1.0)),
        "intentional (eta=0.25)": Intentional(
            cfg=IntentionalConfig(gamma=gamma, trace_lambda=trace_lambda, eta=0.25)
        ),
        "obgd/stream-TD (lr=1.0)": ObGD(cfg=ObGDConfig(lr=1.0, kappa=2.0)),
    })

def mean_step_size(logs, opt):
    key = f"{opt.name}/step_size"
    if key not in logs:
        return float("nan")
    ss = np.asarray(logs[key]).reshape(S, T)
    return ss.mean(), ss[:, : T // 10].mean(), ss[:, -T // 10 :].mean()


print(f"\n=== {env_id}  |  seeds={S}  steps={T:_}  (RMSE vs discounted return, denormalized) ===\n")
print(f"{'optimizer':<26}{'RMSE first':>11}{'RMSE last':>11}{'|TD| last':>11}{'lr mean':>11}{'lr early':>11}{'lr late':>11}{'finite':>8}")
for name, opt in optimizers.items():
    logs, dt = run(opt)
    rf, rl, tf, tl, fin = metrics(logs)
    lr_mean, lr_early, lr_late = mean_step_size(logs, opt)
    print(f"{name:<26}{rf:>11.2f}{rl:>11.2f}{tl:>11.4f}{lr_mean:>11.3e}{lr_early:>11.3e}{lr_late:>11.3e}{str(bool(fin)):>8}")

if not args.compare:
    logs, _ = run(measured)
    ss = np.asarray(logs["measured/step_size"]).reshape(S, T)
    gated = np.asarray(logs["measured/expansive_fraction"]).reshape(S, T)
    m = np.asarray(logs["measured/m_hat"]).reshape(S, T)
    s = np.asarray(logs["measured/s_hat"]).reshape(S, T)
    y = np.asarray(logs["measured/y_hat"]).reshape(S, T)
    n = T // 10
    print(f"\nmeasured diagnostics:")
    print(f"  step size alpha : mean={ss.mean():.3e}  last10%={ss[:, -n:].mean():.3e}  max={ss.max():.3e} (cap={args.eta and measured.cfg.alpha_max})")
    print(f"  gated-off frac  : mean={gated.mean():.3f}  last10%={gated[:, -n:].mean():.3f}")
    print(f"  moments last10% : E[X]={m[:, -n:].mean():.3e}  E[X^2]={s[:, -n:].mean():.3e}  E[d^2|z|^2]={y[:, -n:].mean():.3e}")
