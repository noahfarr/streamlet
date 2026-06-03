"""Profile the example agent configurations with performax to find bottlenecks.

This benchmarks the same agent setups the ``examples/`` scripts use (env +
wrappers + network + optimizer + algorithm) and reports where time goes, using
performax (https://github.com/noahfarr/performax) on top of JAX's Perfetto
tracing.

It reports *device* time per method while the computation runs under
``jax.jit``. Rather than decorate the library, this script instruments each
agent at runtime: every non-dunder method (``train`` and the internals it calls
-- ``_step``, ``_update``, ``_update_step``, the action-selection helpers, ...)
is wrapped with ``performax.scope`` on the instance, which tags the HLO ops with
``jax.named_scope``. Those names survive into the compiled program, so
``performax.device_profile`` can attribute each GPU kernel back to the method
that created it. The library is left untouched.

The train phase is profiled warm (``warmup=True``), so the per-method numbers
are steady-state device time with compilation excluded; XLA compile cost and
steps/sec are measured separately by wall clock.

This is GPU-only and requires CUDA command buffers disabled (handled at import
via ``performax.enable_device_profiling``); see performax/device.py for details.

performax is a separate package, not a streax dependency. Install it first::

    uv pip install -e ~/performax        # or: pip install performax

Examples::

    python scripts/benchmark.py --list
    python scripts/benchmark.py --all
    python scripts/benchmark.py --targets q_lambda_obgd recurrent_q_obgd
"""

import argparse
import inspect
import time
from typing import Callable

try:
    import performax as px
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "performax is not installed in this environment.\n"
        "Install it first, e.g.  uv pip install -e ~/performax"
    ) from exc

# Device profiling needs CUDA command buffers disabled so per-op scope metadata
# reaches the device timeline. The controlling XLA flag is read at backend init,
# so this must run before jax -- and before flax/distrax, which import jax.
px.enable_device_profiling()

import distrax  # noqa: E402
import flax.linen as nn  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from streax.algorithms import (
    ACLambda,
    ACLambdaConfig,
    QLambda,
    QLambdaConfig,
    QRCLambda,
    QRCLambdaConfig,
    RecurrentQLambda,
    RecurrentQLambdaConfig,
    SARSALambda,
    SARSALambdaConfig,
    TDLambda,
    TDLambdaConfig,
)
from streax.environments import environment
from streax.environments.wrappers import (
    NormalizeObservationWrapper,
    NormalizeRewardWrapper,
    RecordEpisodeStatistics,
    StickyActionWrapper,
)
from streax.networks import Flatten, sparse
from streax.optimizers import (
    Calibrated,
    CalibratedConfig,
    Implicit,
    ImplicitConfig,
    ObGD,
    ObGDConfig,
    OptaxOptimizer,
)

# --------------------------------------------------------------------------- #
# Environment / network builders (mirroring the examples)
# --------------------------------------------------------------------------- #


def make_minatar(env_id: str):
    env, env_params = environment.make(env_id)
    env = StickyActionWrapper(env)
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env_params


def make_cartpole():
    env, env_params = environment.make("gymnax::CartPole-v1")
    env = RecordEpisodeStatistics(env)
    env = NormalizeObservationWrapper(env)
    env = NormalizeRewardWrapper(env)
    return env, env_params


def minatar_torso(sparse_init):
    return [
        nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
        Flatten(start_dim=-3),
        nn.Dense(128, kernel_init=sparse_init),
        nn.LayerNorm(),
        nn.leaky_relu,
    ]


def minatar_q_network(num_actions: int):
    si = sparse(sparsity=0.9)
    return nn.Sequential(minatar_torso(si) + [nn.Dense(num_actions, kernel_init=si)])


def minatar_value_network():
    si = sparse(sparsity=0.9)
    return nn.Sequential(minatar_torso(si) + [nn.Dense(1, kernel_init=si)])


def minatar_actor_network(num_actions: int):
    si = sparse(sparsity=0.9)
    return nn.Sequential(
        minatar_torso(si)
        + [
            nn.Dense(num_actions, kernel_init=si),
            lambda logits: distrax.Categorical(logits=logits),
        ]
    )


class RecurrentQNetwork(nn.Module):
    num_actions: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, carry, obs, action, reward, done):
        si = sparse(sparsity=0.9)
        x = nn.Conv(16, (3, 3), strides=(1, 1), padding="VALID", kernel_init=si)(obs)
        x = nn.LayerNorm()(x)
        x = nn.leaky_relu(x)
        x = Flatten(start_dim=-3)(x)
        x = nn.Dense(128, kernel_init=si)(x)
        x = nn.LayerNorm()(x)
        x = nn.leaky_relu(x)
        context = jnp.concatenate(
            [
                jax.nn.one_hot(action, self.num_actions),
                reward[..., None],
                done[..., None].astype(jnp.float32),
            ],
            axis=-1,
        )
        x = jnp.concatenate([x, context], axis=-1)
        carry, hidden = nn.GRUCell(features=self.hidden_size)(carry, x)
        return carry, nn.Dense(self.num_actions, kernel_init=si)(hidden)

    @nn.nowrap
    def initialize_carry(self, rng, num_envs):
        return nn.GRUCell(features=self.hidden_size).initialize_carry(
            rng, (num_envs, 1)
        )


def epsilon_schedule(step):
    return jnp.float32(0.1)


# --------------------------------------------------------------------------- #
# Target registry: name -> builder(args) -> agent
# --------------------------------------------------------------------------- #


def _q(args, optimizer):
    env, env_params = make_minatar(args.env_id)
    n = env.action_space(env_params).n
    cfg = QLambdaConfig(num_envs=args.num_envs, gamma=0.99, trace_lambda=0.8)
    return QLambda(
        cfg, env, env_params, minatar_q_network(n), epsilon_schedule, optimizer
    )


def _sarsa(args, optimizer):
    env, env_params = make_minatar(args.env_id)
    n = env.action_space(env_params).n
    cfg = SARSALambdaConfig(num_envs=args.num_envs, gamma=0.99, trace_lambda=0.8)
    return SARSALambda(
        cfg, env, env_params, minatar_q_network(n), epsilon_schedule, optimizer
    )


def _td(args, optimizer):
    env, env_params = make_cartpole()
    cfg = TDLambdaConfig(num_envs=args.num_envs, gamma=0.99, trace_lambda=0.8)
    return TDLambda(cfg, env, env_params, minatar_value_network(), optimizer)


def _ac(args):
    env, env_params = make_minatar(args.env_id)
    n = env.action_space(env_params).n
    cfg = ACLambdaConfig(
        num_envs=args.num_envs, trace_lambda=0.8, entropy_coefficient=0.01, gamma=0.99
    )
    return ACLambda(
        cfg,
        env,
        env_params,
        minatar_actor_network(n),
        minatar_value_network(),
        ObGD(cfg=ObGDConfig(lr=1.0, kappa=3.0)),
        ObGD(cfg=ObGDConfig(lr=1.0, kappa=2.0)),
    )


def _qrc(args):
    env, env_params = make_minatar(args.env_id)
    n = env.action_space(env_params).n
    cfg = QRCLambdaConfig(
        num_envs=args.num_envs,
        gamma=0.99,
        trace_lambda=0.8,
        gradient_correction=True,
        regularization_coefficient=0.01,
        unroll=2,
    )
    return QRCLambda(
        cfg,
        env,
        env_params,
        minatar_q_network(n),
        minatar_q_network(n),
        OptaxOptimizer(tx=optax.sgd(1e-3), name="q_optimizer"),
        OptaxOptimizer(tx=optax.sgd(1e-3), name="h_optimizer"),
        epsilon_schedule,
    )


def _recurrent_q(args, optimizer):
    env, env_params = make_minatar(args.env_id)
    n = env.action_space(env_params).n
    cfg = RecurrentQLambdaConfig(num_envs=args.num_envs, gamma=0.99, trace_lambda=0.8)
    return RecurrentQLambda(
        cfg,
        env,
        env_params,
        RecurrentQNetwork(num_actions=n),
        epsilon_schedule,
        optimizer,
    )


TARGETS: dict[str, Callable] = {
    "q_lambda_obgd": lambda a: _q(a, ObGD(cfg=ObGDConfig(lr=1.0))),
    "sarsa_lambda_obgd": lambda a: _sarsa(a, ObGD(cfg=ObGDConfig(lr=1.0))),
    "td_lambda_obgd": lambda a: _td(a, ObGD(cfg=ObGDConfig(lr=1.0))),
    "ac_lambda_obgd": _ac,
}


# --------------------------------------------------------------------------- #
# Runtime instrumentation
# --------------------------------------------------------------------------- #


def instrument(agent) -> None:
    cls = type(agent)
    for name in dir(cls):
        if name.startswith("__"):
            continue
        try:
            attr = inspect.getattr_static(cls, name)
        except AttributeError:
            continue
        if not inspect.isfunction(attr):
            continue
        bound = getattr(agent, name)
        setattr(agent, name, px.scope(name=name)(bound))


def profile_target(name: str, build: Callable, args) -> dict:
    agent = build(args)
    instrument(agent)

    seeds, steps = args.seeds, args.steps
    key = jax.random.key(args.seed)

    vinit = jax.jit(jax.vmap(agent.init))

    @jax.jit
    def vtrain(keys, state):
        return jax.vmap(agent.train, in_axes=(0, 0, None))(keys, state, steps)

    state = vinit(jax.random.split(key, seeds))
    jax.block_until_ready(state)
    train_keys = jax.random.split(key, seeds)

    # Wall-clock: cold (first call, includes XLA compilation) and warm.
    t0 = time.perf_counter()
    jax.block_until_ready(vtrain(train_keys, state))
    t_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    jax.block_until_ready(vtrain(train_keys, state))
    t_warm = time.perf_counter() - t0
    compile_ms = max(0.0, t_cold - t_warm) * 1000.0
    sps = int(seeds * steps / t_warm) if t_warm > 0 else 0

    # Per-method device time for the (warm) train phase.
    _, device = px.device_profile(vtrain, warmup=True)(train_keys, state)

    return {"name": name, "device": device, "sps": sps, "compile_ms": compile_ms}


def _scope_ms(result, scope: str) -> float:
    for s in result.stats:
        if s.name == scope:
            return s.total_duration_ms
    return 0.0


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY  (train_device = warm steady-state device ms for the train scope)")
    print("=" * 80)
    header = f"{'target':<26}{'train_device_ms':>16}{'compile_ms':>12}{'steps/sec':>14}"
    print(header)
    print("-" * len(header))
    for r in sorted(
        results, key=lambda r: _scope_ms(r["device"], "train"), reverse=True
    ):
        train = _scope_ms(r["device"], "train")
        print(f"{r['name']:<26}{train:>16.1f}{r['compile_ms']:>12.1f}{r['sps']:>14,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targets", nargs="*", help="Subset of targets to benchmark.")
    parser.add_argument("--all", action="store_true", help="Benchmark every target.")
    parser.add_argument("--list", action="store_true", help="List targets and exit.")
    parser.add_argument("--env-id", default="gymnax::Breakout-MinAtar")
    parser.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Train steps to profile. Steady-state device time converges well "
        "below the default; much higher counts overflow CUPTI's activity buffer "
        "(dropped events under-count device time), so keep this moderate.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--barriers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert optimization barriers at scope boundaries so XLA does not "
        "fuse across methods (required for clean per-method device attribution). "
        "Disable if a pytree return trips the barrier.",
    )
    parser.add_argument("--logger", choices=["console", "rich"], default="console")
    args = parser.parse_args()

    if args.list:
        for name in TARGETS:
            print(name)
        return

    if args.all or not args.targets:
        names = list(TARGETS)
    else:
        names = args.targets
        unknown = [n for n in names if n not in TARGETS]
        if unknown:
            raise SystemExit(f"Unknown targets: {unknown}. Use --list.")

    if args.barriers:
        px.enable_barriers()
    logger = px.RichLogger() if args.logger == "rich" else px.ConsoleLogger()

    results = []
    for name in names:
        print(f"\n### {name} (steps={args.steps:_}, seeds={args.seeds})")
        result = profile_target(name, TARGETS[name], args)
        results.append(result)
        print("-- per-method device time (warm, steady-state) --")
        print(logger.log(result["device"]))
        print(f"throughput: {result['sps']:,} env-steps/sec")

    print_summary(results)


if __name__ == "__main__":
    main()
