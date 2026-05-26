import gymnax
from gymnax.environments import environment


def make(
    env_id: str, **kwargs
) -> tuple[environment.Environment, environment.EnvParams]:
    env, env_params = gymnax.make(env_id, **kwargs)
    return env, env_params
