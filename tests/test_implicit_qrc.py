import jax
import jax.numpy as jnp

from streax.optimizers import Implicit, ImplicitConfig


def make_inputs():
    gradient = {"w": jnp.array([1.0, 0.0])}
    trace = {"w": jnp.array([1.0, 1.0])}
    td_error_grad = {"w": jnp.array([0.5, -0.5])}
    td_error = jnp.asarray(0.4)
    curvature = jnp.asarray(0.5)
    h_value = jnp.asarray(0.3)
    bias_trace = jnp.asarray(0.2)
    return gradient, trace, td_error_grad, td_error, curvature, h_value, bias_trace


def test_qrc_reduces_to_implicit_when_no_correction():
    optimizer = Implicit(cfg=ImplicitConfig(gamma=0.99, trace_lambda=0.8, eta=0.5))
    gradient, trace, b, td_error, curvature, _, _ = make_inputs()

    state = optimizer.init({"w": jnp.zeros((2,))})

    plain_updates, plain_state = optimizer.update(
        state, gradient, trace, td_error, curvature
    )
    qrc_updates, qrc_state = optimizer.update(
        state,
        gradient,
        trace,
        td_error,
        curvature,
        td_error_grad=b,
        h_value=jnp.asarray(0.0),
        bias_trace=jnp.asarray(0.0),
    )

    assert jnp.allclose(plain_updates["w"], qrc_updates["w"], atol=1e-6)
    assert jnp.allclose(plain_state.second_moment["w"], qrc_state.second_moment["w"])


def test_qrc_correction_algebra():
    eta, kappa = 0.5, 1.0
    cfg = ImplicitConfig(
        gamma=0.99,
        trace_lambda=0.8,
        eta=eta,
        kappa=kappa,
        eps=1e-8,
        use_rmsprop=False,
        use_sigma=False,
        use_adaptive_clip=False,
        normalize_delta=False,
    )
    optimizer = Implicit(cfg=cfg)
    gradient, trace, b, td_error, curvature, h_value, bias_trace = make_inputs()

    state = optimizer.init({"w": jnp.zeros((2,))})
    updates, _ = optimizer.update(
        state,
        gradient,
        trace,
        td_error,
        curvature,
        td_error_grad=b,
        h_value=h_value,
        bias_trace=bias_trace,
    )

    g = gradient["w"]
    z = trace["w"]
    bb_vec = b["w"]
    baseline = jnp.sum(z**2)
    denom = jnp.maximum(baseline + eta * curvature, kappa * baseline)
    step = eta / jnp.maximum(denom, cfg.eps)
    base_step = eta / jnp.maximum(baseline, cfg.eps)
    bg = jnp.sum(bb_vec * g)
    bb = jnp.sum(bb_vec * bb_vec)
    safe_delta = jnp.clip(td_error, -1.0, 1.0)
    proximal_delta = safe_delta - base_step * (h_value * bg + bias_trace * bb)
    expected = (
        step * proximal_delta * z
        - base_step * h_value * g
        - base_step * bias_trace * bb_vec
    )

    assert jnp.allclose(updates["w"], expected, atol=1e-6)


def test_qrc_correction_changes_update():
    optimizer = Implicit(cfg=ImplicitConfig(gamma=0.99, trace_lambda=0.8, eta=0.5))
    gradient, trace, b, td_error, curvature, h_value, bias_trace = make_inputs()
    state = optimizer.init({"w": jnp.zeros((2,))})

    no_corr, _ = optimizer.update(
        state,
        gradient,
        trace,
        td_error,
        curvature,
        td_error_grad=b,
        h_value=jnp.asarray(0.0),
        bias_trace=jnp.asarray(0.0),
    )
    with_corr, _ = optimizer.update(
        state,
        gradient,
        trace,
        td_error,
        curvature,
        td_error_grad=b,
        h_value=h_value,
        bias_trace=bias_trace,
    )

    assert not jnp.allclose(no_corr["w"], with_corr["w"])


def test_qrc_update_is_jittable():
    optimizer = Implicit(cfg=ImplicitConfig(gamma=0.99, trace_lambda=0.8))
    gradient = {"w": jnp.ones((3,))}
    trace = {"w": jnp.ones((3,))}
    b = {"w": jnp.full((3,), 0.5)}
    td_error = jnp.asarray(0.5)
    curvature = jnp.asarray(1.0)
    h_value = jnp.asarray(0.1)
    bias_trace = jnp.asarray(0.3)
    state = optimizer.init({"w": jnp.ones((3,))})

    updates, new_state = jax.jit(optimizer.update)(
        state,
        gradient,
        trace,
        td_error,
        curvature,
        td_error_grad=b,
        h_value=h_value,
        bias_trace=bias_trace,
    )
    assert updates["w"].shape == (3,)
    assert new_state.second_moment["w"].shape == (3,)


if __name__ == "__main__":
    test_qrc_reduces_to_implicit_when_no_correction()
    test_qrc_correction_algebra()
    test_qrc_correction_changes_update()
    test_qrc_update_is_jittable()
    print("all passed")
