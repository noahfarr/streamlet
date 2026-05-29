import flax.linen as nn
import jax
import jax.numpy as jnp

from stremax.optimizers import VOGD, VOGDConfig


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
    cfg = VOGDConfig(eta=eta, beta=0.05, s0=1e-8, alpha_max=1e9)
    optimizer = VOGD(cfg=cfg)

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


def test_update_is_jittable():
    cfg = VOGDConfig()
    optimizer = VOGD(cfg=cfg)
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
    test_update_is_jittable()
    print("all passed")
