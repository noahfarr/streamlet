from streax.environments import ale, brax, ett, gymnasium, gymnax

register = {
    "gymnax": gymnax.make,
    "brax": brax.make,
    "gymnasium": gymnasium.make,
    "ale": ale.make,
    "ett": ett.make,
}


def make(
    env_id,
    **kwargs,
) -> tuple:
    namespace, env_id = env_id.split("::", 1)

    if namespace not in register:
        raise ValueError(f"Unknown namespace {namespace}")

    env, env_params = register[namespace](env_id, **kwargs)

    return env, env_params
