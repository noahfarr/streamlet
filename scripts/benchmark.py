"""Profile the example agent configurations with performax to find bottlenecks.

This benchmarks the same agent setups the ``examples/`` scripts use (env +
wrappers + network + optimizer + algorithm) and reports where time goes, using
performax (https://github.com/noahfarr/performax) on top of JAX's Perfetto
tracing.

performax only measures regions wrapped with ``@track``. Rather than decorate
the library, this script instruments each agent at runtime: every non-dunder
method (``init``, ``train``, ``evaluate`` and the internals they call --
``_step``, ``_update``, ``_update_step``, the action-selection helpers, ...) is
wrapped with ``track`` on the instance. The library is left untouched.

Each target is profiled twice: cold (first call, includes XLA compilation) and
warm (``warmup=True``, steady-state). steps/sec is reported for the train phase.

performax is a separate package, not a streax dependency. Install it first::

    uv pip install -e ~/performax        # or: pip install performax

Examples::

    python scripts/benchmark.py --list
    python scripts/benchmark.py --all --steps 50000
    python scripts/benchmark.py --targets q_lambda_obgd recurrent_q_obgd
"""

import argparse
import inspect
import time
from typing import Callable

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

try:
    import performax as px
except ModuleNotFoundError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "performax is not installed in this environment.\n"
        "Install it first, e.g.  uv pip install -e ~/performax"
    ) from exc

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
        return nn.GRUCell(features=self.hidden_size).initialize_carry(rng, (num_envs, 1))


def epsilon_schedule(step):
    return jnp.float32(0.1)


# --------------------------------------------------------------------------- #
# Target registry: name -> builder(args) -> agent
# --------------------------------------------------------------------------- #


def _q(args, optimizer):
    env, env_params = make_minatar(args.env_id)
    n = env.action_space(env_params).n
    cfg = QLambdaConfig(num_envs=args.num_envs, gamma=0.99, trace_lambda=0.8)
    return QLambda(cfg, env, env_params, minatar_q_network(n), epsilon_schedule, optimizer)


def _sarsa(args, optimizer):
    env, env_params = make_minatar(args.env_id)
    n = env.action_space(env_params).n
    cfg = SARSALambdaConfig(num_envs=args.num_envs, gamma=0.99, trace_lambda=0.8)
    return SARSALambda(cfg, env, env_params, minatar_q_network(n), epsilon_schedule, optimizer)


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
        cfg, env, env_params, RecurrentQNetwork(num_actions=n), epsilon_schedule, optimizer
    )


TARGETS: dict[str, Callable] = {
    "q_lambda_obgd": lambda a: _q(a, ObGD(cfg=ObGDConfig(lr=1.0))),
    "q_lambda_obgd_exact": lambda a: _q(a, ObGD(cfg=ObGDConfig(lr=1e-3, exact=True))),
    "q_lambda_calibrated": lambda a: _q(a, Calibrated(cfg=CalibratedConfig())),
    "sarsa_lambda_obgd": lambda a: _sarsa(a, ObGD(cfg=ObGDConfig(lr=1.0))),
    "sarsa_lambda_implicit": lambda a: _sarsa(a, Implicit(cfg=ImplicitConfig(lr=1.0))),
    "td_lambda_obgd": lambda a: _td(a, ObGD(cfg=ObGDConfig(lr=1.0))),
    "ac_lambda_obgd": _ac,
    "qrc": _qrc,
    "recurrent_q_obgd": lambda a: _recurrent_q(a, ObGD(cfg=ObGDConfig(lr=1.0))),
}


# --------------------------------------------------------------------------- #
# Runtime instrumentation
# --------------------------------------------------------------------------- #


def track_methods(agent) -> None:
    """Wrap every non-dunder method of ``agent`` with performax ``track``.

    Instance-only: replaces each bound method with a tracked version so the
    library is untouched. Names are the method names (``train``, ``_step``,
    ``_update``, ...), which is what shows up in the profile.
    """
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
        setattr(agent, name, px.track(name=name)(bound))


def _leaves(state):
    # Return a tuple of array leaves so performax blocks on them (a struct
    # dataclass has no block_until_ready; px.profile only blocks arrays/tuples).
    return tuple(jax.tree_util.tree_leaves(state))


def profile_target(name: str, build: Callable, args) -> dict:
    agent = build(args)
    track_methods(agent)

    seeds, steps = args.seeds, args.steps
    key = jax.random.key(args.seed)

    vinit = jax.jit(jax.vmap(agent.init))

    @jax.jit
    def vtrain(keys, state):
        return jax.vmap(agent.train, in_axes=(0, 0, None))(keys, state, steps)

    has_eval = hasattr(agent, "evaluate")

    @jax.jit
    def veval(keys, state):
        return jax.vmap(agent.evaluate, in_axes=(0, 0, None))(keys, state, steps)

    def run(key):
        ik, tk, ek = jax.random.split(key, 3)
        state = vinit(jax.random.split(ik, seeds))
        state = vtrain(jax.random.split(tk, seeds), state)
        if has_eval:
            state = veval(jax.random.split(ek, seeds), state)
        return _leaves(state)

    # Cold first (nothing compiled yet) -> includes XLA compilation.
    _, cold = px.profile(run)(key)
    # Warm -> steady-state device time only.
    _, warm = px.profile(run, warmup=True)(key)

    # Wall-clock steps/sec for the (already compiled) train phase.
    state = vinit(jax.random.split(key, seeds))
    jax.block_until_ready(state)
    state = vtrain(jax.random.split(key, seeds), state)  # ensure compiled
    jax.block_until_ready(state)
    t0 = time.perf_counter()
    state = vtrain(jax.random.split(key, seeds), state)
    jax.block_until_ready(state)
    dt = time.perf_counter() - t0
    sps = int(seeds * steps / dt) if dt > 0 else 0

    return {"name": name, "cold": cold, "warm": warm, "sps": sps}


def _phase(result, phase: str) -> float:
    for s in result.stats:
        if s.name == phase:
            return s.total_duration_ms
    return 0.0


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("SUMMARY  (warm = steady-state device ms; compile = cold - warm, train phase)")
    print("=" * 80)
    header = f"{'target':<26}{'train_warm':>12}{'train_compile':>15}{'steps/sec':>14}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: _phase(r["warm"], "train"), reverse=True):
        warm = _phase(r["warm"], "train")
        compile_ms = max(0.0, _phase(r["cold"], "train") - warm)
        print(f"{r['name']:<26}{warm:>12.1f}{compile_ms:>15.1f}{r['sps']:>14,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--targets", nargs="*", help="Subset of targets to benchmark.")
    parser.add_argument("--all", action="store_true", help="Benchmark every target.")
    parser.add_argument("--list", action="store_true", help="List targets and exit.")
    parser.add_argument("--env-id", default="gymnax::Breakout-MinAtar")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--barriers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Insert optimization barriers at tracked boundaries (sharper "
        "per-region timing). Disable if a pytree return trips the barrier.",
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
        print("-- warm (steady-state device time per method) --")
        print(logger.log(result["warm"]))
        print(f"throughput: {result['sps']:,} env-steps/sec")

    print_summary(results)


if __name__ == "__main__":
    main()
