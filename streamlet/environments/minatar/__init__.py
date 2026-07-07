from streamlet.environments.minatar.asterix import Asterix
from streamlet.environments.minatar.breakout import Breakout
from streamlet.environments.minatar.freeway import Freeway
from streamlet.environments.minatar.seaquest import Seaquest
from streamlet.environments.minatar.space_invaders import SpaceInvaders

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
