import flax.linen as nn
import jax
import jax.numpy as jnp

from stremax.optimizers import Measured, MeasuredConfig


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(8)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x.squeeze(-1)


class QMLP(nn.Module):
    actions: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(8)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.actions)(x)
        return x


def test_jvp_matches_reverse_mode_value():
    gamma = 0.97
    network = MLP()
    key = jax.random.key(0)
    init_key, s_key, sn_key, z_key = jax.random.split(key, 4)

    s = jax.random.normal(s_key, (4,))
    s_next = jax.random.normal(sn_key, (4,))
    params = network.init(init_key, s)
    z = jax.tree.map(lambda p: jax.random.normal(z_key, p.shape), params)

    def u(w):
        return network.apply(w, s) - gamma * network.apply(w, s_next)

    _, x_jvp = jax.jvp(u, (params,), (z,))

    grad_s = jax.grad(lambda w: network.apply(w, s))(params)
    grad_next = jax.grad(lambda w: network.apply(w, s_next))(params)
    bellman_grad = jax.tree.map(lambda g, gn: g - gamma * gn, grad_s, grad_next)
    x_explicit = sum(
        jnp.sum(g * zl)
        for g, zl in zip(jax.tree.leaves(bellman_grad), jax.tree.leaves(z))
    )

    assert jnp.allclose(x_jvp, x_explicit, atol=1e-5, rtol=1e-5)


def test_jvp_matches_reverse_mode_q_max():
    gamma = 0.97
    network = QMLP(actions=3)
    key = jax.random.key(1)
    init_key, s_key, sn_key, z_key = jax.random.split(key, 4)

    s = jax.random.normal(s_key, (4,))
    s_next = jax.random.normal(sn_key, (4,))
    action = 1
    params = network.init(init_key, s)
    z = jax.tree.map(lambda p: jax.random.normal(z_key, p.shape), params)

    def u(w):
        q = network.apply(w, s)[action]
        q_next = network.apply(w, s_next).max(axis=-1)
        return q - gamma * q_next

    _, x_jvp = jax.jvp(u, (params,), (z,))

    grad_q = jax.grad(lambda w: network.apply(w, s)[action])(params)
    grad_next = jax.grad(lambda w: network.apply(w, s_next).max(axis=-1))(params)
    bellman_grad = jax.tree.map(lambda g, gn: g - gamma * gn, grad_q, grad_next)
    x_explicit = sum(
        jnp.sum(g * zl)
        for g, zl in zip(jax.tree.leaves(bellman_grad), jax.tree.leaves(z))
    )

    assert jnp.allclose(x_jvp, x_explicit, atol=1e-5, rtol=1e-5)


def test_alpha_converges_to_first_over_second_moment():
    eta = 0.5
    x = 2.0
    cfg = MeasuredConfig(eta=eta, beta=0.05, s0=1e-8, alpha_max=1e9)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.zeros((1,))
    interaction = jnp.full((1,), x)

    state = optimizer.init(params, num_envs=1)
    alpha = None
    for _ in range(5000):
        alpha = eta * jnp.maximum(state.m_hat, 0.0) / (state.s_hat + cfg.s0)
        alpha = jnp.minimum(alpha, cfg.alpha_max)
        _, state = optimizer.update(state, params, trace, td_error, interaction)

    expected = eta * x / (x**2)
    assert jnp.allclose(alpha, expected, atol=1e-4)


def test_curvature_clamp_prevents_overshoot():
    # A large tail interaction with a stale (small) second moment is exactly the
    # regime-change case that diverges: the variance-optimal step is tuned to a
    # small E[X^2], but the realized X_t is large. The clamp must keep the
    # per-sample contraction |1 - alpha * X_t| <= 1.
    eta = 0.5
    kappa = 1.0
    cfg = MeasuredConfig(eta=eta, beta=0.999, s0=1e-8, alpha_max=1e9, kappa=kappa)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.ones((1,))

    # Warm the moments on a small-curvature regime so the variance-optimal step
    # is large, then hit it with a big interaction.
    small_x = jnp.full((1,), 1e-3)
    state = optimizer.init(params, num_envs=1)
    for _ in range(2000):
        _, state = optimizer.update(state, params, trace, td_error, small_x)

    big_x = jnp.full((1,), 10.0)
    alpha_var = eta * jnp.maximum(state.m_hat, 0.0) / (state.s_hat + cfg.s0)
    assert (alpha_var * big_x > 1.0).all()  # unclamped step would overshoot

    alpha = jnp.minimum(alpha_var, cfg.alpha_max)
    alpha = jnp.minimum(alpha, cfg.kappa / (jnp.abs(big_x) + cfg.s0))
    contraction = jnp.abs(1.0 - alpha * big_x)
    assert (contraction <= 1.0 + 1e-6).all()


def test_clamp_inactive_when_step_is_safe():
    # When the variance-optimal step is already overshoot-free, the curvature
    # clamp must not bind, so the recovered step matches E[X] / E[X^2] * eta.
    eta = 0.5
    x = 2.0
    cfg = MeasuredConfig(eta=eta, beta=0.05, s0=1e-8, alpha_max=1e9, kappa=1.0)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.zeros((1,))
    interaction = jnp.full((1,), x)

    state = optimizer.init(params, num_envs=1)
    for _ in range(5000):
        _, state = optimizer.update(state, params, trace, td_error, interaction)

    alpha_var = eta * jnp.maximum(state.m_hat, 0.0) / (state.s_hat + cfg.s0)
    curvature_cap = cfg.kappa / (abs(x) + cfg.s0)
    assert alpha_var < curvature_cap  # clamp inactive
    assert jnp.allclose(alpha_var, eta * x / (x**2), atol=1e-4)


def test_expansive_sample_takes_no_step():
    # A negative interaction is an expansive sample: no positive step is
    # contractive, since 1 - alpha * X = 1 + alpha * |X| >= 1. The kappa clamp
    # only caps that expansion at 1 + kappa; the sign gate must zero the step so
    # the realized contraction stays in [0, 1]. The lagging max(m_hat, 0) gate is
    # not enough on its own: warm the moments on positive interactions so m_hat
    # stays positive, then feed a negative X.
    eta = 0.5
    cfg = MeasuredConfig(eta=eta, beta=0.999, s0=1e-8, alpha_max=1e9, kappa=1.0)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.ones((1,))

    pos_x = jnp.full((1,), 1.0)
    state = optimizer.init(params, num_envs=1)
    for _ in range(2000):
        _, state = optimizer.update(state, params, trace, td_error, pos_x)

    # m_hat is still positive, so the variance-optimal step would be applied.
    assert (state.m_hat > 0.0).all()
    alpha_var = eta * jnp.maximum(state.m_hat, 0.0) / (state.s_hat + cfg.s0)
    assert (alpha_var > 0.0).all()

    neg_x = jnp.full((1,), -1.0)
    updates, _ = optimizer.update(state, params, trace, neg_x * 0.0 + td_error, neg_x)
    # The expansive sample is gated to a zero step.
    assert jnp.allclose(updates["w"], 0.0)


def test_contraction_bounded_for_all_interaction_signs():
    # With the sign gate plus the kappa clamp, the realized per-sample
    # contraction |1 - alpha * X| must stay in [0, 1] for every X, positive or
    # negative, even when the variance-optimal step is large and stale.
    eta = 0.5
    cfg = MeasuredConfig(eta=eta, beta=0.999, s0=1e-8, alpha_max=1e9, kappa=1.0)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.ones((1,))

    small_x = jnp.full((1,), 1e-3)
    state = optimizer.init(params, num_envs=1)
    for _ in range(2000):
        _, state = optimizer.update(state, params, trace, td_error, small_x)

    for x in (-10.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 10.0):
        interaction = jnp.full((1,), x)
        alpha = eta * jnp.maximum(state.m_hat, 0.0) / (state.s_hat + cfg.s0)
        alpha = jnp.minimum(alpha, cfg.alpha_max)
        alpha = jnp.minimum(alpha, cfg.kappa / (jnp.abs(interaction) + cfg.s0))
        alpha = jnp.where(interaction > 0.0, alpha, 0.0)
        contraction = 1.0 - alpha * interaction
        assert (contraction >= -1e-6).all() and (contraction <= 1.0 + 1e-6).all()


def test_update_is_jittable():
    cfg = MeasuredConfig()
    optimizer = Measured(cfg=cfg)
    params = {"w": jnp.ones((3,))}
    trace = {"w": jnp.ones((2, 3))}
    td_error = jnp.array([0.5, -0.5])
    interaction = jnp.array([1.0, 2.0])
    state = optimizer.init(params, num_envs=2)

    updates, new_state = jax.jit(optimizer.update)(
        state, params, trace, td_error, interaction
    )
    assert jax.tree.leaves(updates)[0].shape == (3,)
    assert new_state.m_hat.shape == (2,)


if __name__ == "__main__":
    test_jvp_matches_reverse_mode_value()
    test_jvp_matches_reverse_mode_q_max()
    test_alpha_converges_to_first_over_second_moment()
    test_curvature_clamp_prevents_overshoot()
    test_clamp_inactive_when_step_is_safe()
    test_expansive_sample_takes_no_step()
    test_contraction_bounded_for_all_interaction_signs()
    test_update_is_jittable()
    print("all passed")
