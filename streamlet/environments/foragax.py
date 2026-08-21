from foragax.registry import make as make_foragax
from gymnax.environments import environment


def make(
    env_id: str, **kwargs
) -> tuple[environment.Environment, environment.EnvParams]:
    env = make_foragax(env_id, **kwargs)
    return env, env.default_params
