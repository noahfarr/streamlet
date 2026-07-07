import animax
from gymnax.environments import environment


def make(
    env_id: str, **kwargs
) -> tuple[environment.Environment, environment.EnvParams]:
    env, env_params = animax.make(env_id, **kwargs)
    return env, env_params
