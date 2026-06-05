from streax.environments.minatar.asterix import Asterix
from streax.environments.minatar.breakout import Breakout
from streax.environments.minatar.freeway import Freeway
from streax.environments.minatar.seaquest import Seaquest
from streax.environments.minatar.space_invaders import SpaceInvaders

_envs = {
    "Asterix-MinAtar": Asterix,
    "Breakout-MinAtar": Breakout,
    "Freeway-MinAtar": Freeway,
    "Seaquest-MinAtar": Seaquest,
    "SpaceInvaders-MinAtar": SpaceInvaders,
}


def make(env_id: str, **kwargs):
    if env_id not in _envs:
        raise ValueError(f"Unknown minatar environment {env_id}")
    env = _envs[env_id](**kwargs)
    return env, env.default_params


__all__ = [
    "Asterix",
    "Breakout",
    "Freeway",
    "Seaquest",
    "SpaceInvaders",
    "make",
]
