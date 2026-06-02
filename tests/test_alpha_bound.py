import jax
import jax.numpy as jnp

from streax.optimizers import AlphaBound, AlphaBoundConfig


def alpha_from_update(updates):
    # With unit trace and unit TD error the applied update equals the step size.
    return float(updates["w"][0])


def feed(optimizer, interactions, td_error=1.0):
    params = {"w": jnp.zeros((1,))}
    trace = {"w": jnp.ones((1, 1))}
    state = optimizer.init(params, num_envs=1)
    updates = None
    for x in interactions:
        updates, state = optimizer.update(
            state, params, trace, jnp.full((1,), td_error), jnp.full((1,), x)
        )
    return updates, state


def test_init_alpha_is_one():
    optimizer = AlphaBound(cfg=AlphaBoundConfig())
    state = optimizer.init({"w": jnp.zeros((1,))}, num_envs=1)
    assert jnp.allclose(state.alpha, 1.0)


def test_bound_caps_step_at_inverse_interaction():
    # A large interaction caps the step at 1/|X|; a small one leaves alpha at the init.
    big, _ = feed(AlphaBound(cfg=AlphaBoundConfig()), [4.0])
    assert jnp.allclose(alpha_from_update(big), 1.0 / 4.0, atol=1e-6)

    small, _ = feed(AlphaBound(cfg=AlphaBoundConfig()), [0.5])
    assert jnp.allclose(alpha_from_update(small), 1.0, atol=1e-6)  # min(1, 2) = 1


def test_step_only_decreases_running_minimum():
    # alpha is a running minimum: a later small interaction never raises it back.
    _, state = feed(AlphaBound(cfg=AlphaBoundConfig()), [2.0, 8.0, 1.0, 0.1])
    # min over 1/|X|: 1/2, 1/8, 1, 10 -> 1/8, and the init 1.0 -> overall 1/8.
    assert jnp.allclose(state.alpha, 1.0 / 8.0, atol=1e-6)


def test_absolute_value_of_interaction():
    # The bound uses |X|, so a negative interaction caps the step just like its magnitude.
    neg, _ = feed(AlphaBound(cfg=AlphaBoundConfig()), [-4.0])
    assert jnp.allclose(alpha_from_update(neg), 1.0 / 4.0, atol=1e-6)


def test_update_equals_alpha_times_td_error_times_trace():
    optimizer = AlphaBound(cfg=AlphaBoundConfig())
    params = {"w": jnp.zeros((2,))}
    trace = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    td_error = jnp.array([0.5, -1.0])
    interaction = jnp.array([4.0, 2.0])

    state = optimizer.init(params, num_envs=2)
    updates, state = optimizer.update(state, params, trace, td_error, interaction)

    alpha = jnp.minimum(1.0, 1.0 / jnp.abs(interaction))
    expected = (alpha[:, None] * td_error[:, None] * trace["w"]).mean(axis=0)
    assert jnp.allclose(updates["w"], expected, atol=1e-6)


def test_update_is_jittable():
    optimizer = AlphaBound(cfg=AlphaBoundConfig())
    params = {"w": jnp.ones((3,))}
    trace = {"w": jnp.ones((2, 3))}
    td_error = jnp.array([0.5, -0.5])
    interaction = jnp.array([1.0, 2.0])
    state = optimizer.init(params, num_envs=2)

    updates, new_state = jax.jit(optimizer.update)(
        state, params, trace, td_error, interaction
    )
    assert jax.tree.leaves(updates)[0].shape == (3,)
    assert new_state.alpha.shape == (2,)


if __name__ == "__main__":
    test_init_alpha_is_one()
    test_bound_caps_step_at_inverse_interaction()
    test_step_only_decreases_running_minimum()
    test_absolute_value_of_interaction()
    test_update_equals_alpha_times_td_error_times_trace()
    test_update_is_jittable()
    print("all passed")
