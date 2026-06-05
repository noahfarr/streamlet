import sys
from pathlib import Path

import jax
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import rarespike_analysis as ra  # noqa: E402

from streax.algorithms import TDLambda, TDLambdaConfig  # noqa: E402
from streax.environments import environment  # noqa: E402
from streax.environments.rarespike import (  # noqa: E402
    RareSpike,
    RareSpikeParams,
    RareSpikeState,
)
from streax.optimizers import AlphaBound, AlphaBoundConfig  # noqa: E402

CASES = [(3.0, 0.03, 0.0), (3.0, 0.03, 0.9), (5.0, 0.05, 0.0), (2.0, 0.1, 0.5)]


def test_mean_step_rho_equals_cv2():
    # rho at the mean-stability-centred step alpha = 1/A is exactly CV^2(X).
    for M, p, gamma in CASES:
        alpha = ra.alpha_mean_step(M, p, gamma)
        assert jnp.allclose(ra.rho(alpha, M, p, gamma), ra.cv2(M, p, gamma), atol=1e-6)


def test_calibrated_rho_equals_cv2_over_one_plus_cv2():
    # rho at the variance-optimal step alpha* = A/E[X^2] is CV^2/(1+CV^2) < 1.
    for M, p, gamma in CASES:
        alpha = ra.alpha_calibrated(M, p, gamma)
        c = ra.cv2(M, p, gamma)
        assert jnp.allclose(ra.rho(alpha, M, p, gamma), c / (1.0 + c), atol=1e-6)
        assert ra.rho(alpha, M, p, gamma) < 1.0


def test_threshold_ratio_is_one_plus_cv2():
    # The mean vs mean-square stability thresholds differ by exactly 1 + CV^2(X).
    for M, p, gamma in CASES:
        ratio = ra.mean_threshold(M, p, gamma) / ra.ms_threshold(M, p, gamma)
        assert jnp.allclose(ratio, 1.0 + ra.cv2(M, p, gamma), atol=1e-6)


def test_concrete_numbers():
    # The construction's headline instance: M=3, p=0.03, gamma=0.
    s = ra.stability(M=3.0, p=0.03, gamma=0.0)
    assert jnp.allclose(s.A, 1.24, atol=1e-6)
    assert jnp.allclose(s.E_X2, 3.40, atol=1e-6)
    assert jnp.allclose(s.cv2, 1.2112, atol=1e-3)
    assert jnp.allclose(s.rho_mean_step, 1.2112, atol=1e-3)
    assert jnp.allclose(s.rho_calibrated, 0.5478, atol=1e-3)
    assert s.rho_mean_step > 1.0  # the mean step 1/A diverges in mean square
    assert s.rho_calibrated < 1.0  # Calibrated contracts


def test_cv2_crosses_one_with_spike_magnitude():
    # The phase transition: CV^2 crosses 1 as the spike magnitude grows.
    assert ra.cv2(2.0, 0.03, 0.0) < 1.0
    assert ra.cv2(5.0, 0.03, 0.0) > 1.0


def test_env_feature_distribution_matches_moments():
    M, p = 3.0, 0.03
    env, params = environment.make("rarespike::Spike", M=M, p=p)
    assert isinstance(env, RareSpike)
    assert isinstance(params, RareSpikeParams)

    state0 = RareSpikeState(step=0)
    keys = jax.random.split(jax.random.key(0), 200_000)
    obs = jax.vmap(lambda k: env.step(k, state0, 0, params)[0])(keys).reshape(-1)

    assert obs.shape[0] == 200_000
    assert jnp.all((obs == 1.0) | (obs == M))  # only the two feature values appear
    assert jnp.allclose(obs.mean(), ra.feature_moment(1, M, p), atol=2e-3)
    assert jnp.allclose((obs**2).mean(), ra.feature_moment(2, M, p), atol=2e-2)


def test_env_reward_is_zero_and_not_done():
    env, params = environment.make("rarespike::Spike")
    obs, state = env.reset(jax.random.key(0), params)
    assert obs.shape == (1,)
    _, _, reward, done, _ = env.step(jax.random.key(1), state, 0, params)
    assert float(reward) == 0.0
    assert bool(done) is False


def test_alphabound_integrates_with_tdlambda():
    # Smoke test: AlphaBound is routed through the curvature branch and trains.
    import flax.linen as nn
    import lox

    env, params = environment.make("rarespike::Spike", M=3.0, p=0.03)
    config = TDLambdaConfig(gamma=0.0, trace_lambda=0.0)
    network = nn.Dense(1, use_bias=False, kernel_init=nn.initializers.constant(1.0))
    agent = TDLambda(config, env, params, network, AlphaBound(cfg=AlphaBoundConfig()))

    init = jax.vmap(agent.init)
    train = jax.vmap(lox.spool(agent.train), in_axes=(0, 0, None))
    state = init(jax.random.split(jax.random.key(0), 4))
    state, logs = train(jax.random.split(jax.random.key(1), 4), state, 5)

    assert jnp.all(jnp.isfinite(jnp.asarray(logs["alpha_bound/step_size"])))


if __name__ == "__main__":
    test_mean_step_rho_equals_cv2()
    test_calibrated_rho_equals_cv2_over_one_plus_cv2()
    test_threshold_ratio_is_one_plus_cv2()
    test_concrete_numbers()
    test_cv2_crosses_one_with_spike_magnitude()
    test_env_feature_distribution_matches_moments()
    test_env_reward_is_zero_and_not_done()
    test_alphabound_integrates_with_tdlambda()
    print("all passed")
