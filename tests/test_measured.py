import flax.linen as nn
import jax
import jax.numpy as jnp

from streax.optimizers import Measured, MeasuredConfig


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


def measured_alpha(cfg, state):
    # Mirror the step size the optimizer applies: the variance-optimal step of
    # Eq. (7), eta * max(0, E[X]) / (E[X^2] + nu * E[delta^2 ||z||^2]), capped at 1.
    alpha = (
        cfg.eta
        * jnp.maximum(state.m_hat, 0.0)
        / (state.s_hat + cfg.nu * state.y_hat + cfg.eps)
    )
    return jnp.minimum(alpha, cfg.alpha_max)


def test_alpha_converges_to_first_over_second_moment():
    # With a zero TD error the noise term nu * E[delta^2 ||z||^2] vanishes, so
    # the step must converge to the noise-free optimum eta * E[X] / E[X^2].
    eta = 0.5
    x = 2.0
    cfg = MeasuredConfig(eta=eta, beta=0.05)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.zeros((1,))
    interaction = jnp.full((1,), x)

    state = optimizer.init(params, num_envs=1)
    for _ in range(5000):
        _, state = optimizer.update(state, params, trace, td_error, interaction)

    expected = eta * x / (x**2)
    assert jnp.allclose(measured_alpha(cfg, state), expected, atol=1e-4)


def test_target_variance_term_shrinks_step():
    # The defining piece of Eq. (7): the nu * E[delta^2 ||z||^2] noise term in
    # the denominator. A nonzero TD error must drive the step strictly below the
    # noise-free optimum, and the realized denominator must match E[X^2] + nu * y.
    eta = 0.5
    x = 2.0
    delta = 1.5
    nu = 0.1
    cfg = MeasuredConfig(eta=eta, beta=0.05, nu=nu)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}  # ||z||^2 = 3
    z_sq = 3.0
    td_error = jnp.full((1,), delta)
    interaction = jnp.full((1,), x)

    state = optimizer.init(params, num_envs=1)
    for _ in range(5000):
        _, state = optimizer.update(state, params, trace, td_error, interaction)

    assert jnp.allclose(state.m_hat, x, atol=1e-4)
    assert jnp.allclose(state.s_hat, x**2, atol=1e-4)
    assert jnp.allclose(state.y_hat, (delta**2) * z_sq, atol=1e-4)

    expected = eta * x / (x**2 + nu * (delta**2) * z_sq)
    alpha = measured_alpha(cfg, state)
    assert jnp.allclose(alpha, expected, atol=1e-4)
    assert alpha < eta * x / (x**2)  # noise term strictly shrinks the step


def test_nonpositive_mean_interaction_gates_step_off():
    # The paper's one-sided test: when the running estimate of E[X] is
    # non-positive, no positive scalar step contracts, so the step turns off.
    # The gate is on the mean m_hat, not the instantaneous sample.
    eta = 0.5
    cfg = MeasuredConfig(eta=eta, beta=0.05)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.ones((1,))
    neg_x = jnp.full((1,), -1.0)

    state = optimizer.init(params, num_envs=1)
    for _ in range(2000):
        updates, state = optimizer.update(state, params, trace, td_error, neg_x)

    assert (state.m_hat < 0.0).all()
    assert jnp.allclose(measured_alpha(cfg, state), 0.0)
    assert jnp.allclose(updates["w"], 0.0)


def test_step_is_capped_at_one():
    # When the variance-optimal step would exceed 1, the optimizer caps it. With
    # a unit trace and unit TD error the applied update equals the (capped) step,
    # so we can read it straight off the update.
    eta = 1.0
    x = 0.5
    cfg = MeasuredConfig(eta=eta, beta=0.05, nu=0.0)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((3,))}
    trace = {"w": jnp.ones((1, 3))}
    td_error = jnp.ones((1,))
    interaction = jnp.full((1,), x)

    state = optimizer.init(params, num_envs=1)
    updates = None
    for _ in range(5000):
        updates, state = optimizer.update(state, params, trace, td_error, interaction)

    uncapped = eta * x / (x**2)
    assert uncapped > 1.0  # without the cap the step would overshoot
    assert jnp.allclose(measured_alpha(cfg, state), 1.0)
    # update = mean over envs of alpha * td_error * z = 1.0 * 1.0 * 1.0
    assert jnp.allclose(updates["w"], 1.0, atol=1e-4)


def test_update_equals_alpha_times_td_error_times_trace():
    # The applied update is the mean over the env axis of alpha * delta * z.
    eta = 0.3
    cfg = MeasuredConfig(eta=eta, beta=0.05, nu=0.0)
    optimizer = Measured(cfg=cfg)

    params = {"w": jnp.zeros((2,))}
    trace = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    td_error = jnp.array([0.5, -1.0])
    interaction = jnp.array([2.0, 2.0])

    state = optimizer.init(params, num_envs=2)
    updates, _ = optimizer.update(state, params, trace, td_error, interaction)

    # On the first step the moments are still zero, so the gated step is zero.
    assert jnp.allclose(updates["w"], 0.0)

    for _ in range(5000):
        updates, state = optimizer.update(state, params, trace, td_error, interaction)

    alpha = measured_alpha(cfg, state)
    expected = (
        alpha[:, None] * td_error[:, None] * trace["w"]
    ).mean(axis=0)
    assert jnp.allclose(updates["w"], expected, atol=1e-4)


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
    test_target_variance_term_shrinks_step()
    test_nonpositive_mean_interaction_gates_step_off()
    test_step_is_capped_at_one()
    test_update_equals_alpha_times_td_error_times_trace()
    test_update_is_jittable()
    print("all passed")
