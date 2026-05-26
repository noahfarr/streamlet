import lox
import optax


def inject_logger(optimizer_fn, prefix: str = "optimizer"):
    def factory(**kwargs):
        base = optax.inject_hyperparams(optimizer_fn)(**kwargs)

        def update_fn(updates, state, params=None):
            lox.log({f"{prefix}/{k}": v for k, v in state.hyperparams.items()})
            return base.update(updates, state, params)

        return optax.GradientTransformation(init=base.init, update=update_fn)

    return factory
