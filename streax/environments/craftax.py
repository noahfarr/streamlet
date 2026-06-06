def make(env_id: str, auto_reset: bool = True, **kwargs) -> tuple:
    from craftax.craftax_env import make_craftax_env_from_name

    env = make_craftax_env_from_name(env_id, auto_reset=auto_reset, **kwargs)
    env_params = env.default_params
    return env, env_params
