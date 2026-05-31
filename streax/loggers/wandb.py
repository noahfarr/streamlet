import dataclasses
import hashlib

import jax
import wandb

from streax.utils.axes import ensure_axis
from streax.utils.typing import PyTree


def config_str(cfg, prefix="optimizer/"):
    """Concatenate the `prefix`-scoped config fields of `cfg` into one string.

    Examples build the wandb config with the optimizer's dataclass fields under
    `optimizer/<field>` keys (and `actor_optimizer/`, `critic_optimizer/` for the
    actor-critic examples). Collect those into a single stable, sorted string so
    runs can be grouped/compared by their full optimizer configuration.
    """
    items = sorted((k, v) for k, v in cfg.items() if prefix in k)
    return ",".join(f"{k}={v}" for k, v in items)


def config_group(algorithm, env_id, algorithm_cfg, optimizer):
    payload = "|".join(
        [
            algorithm,
            env_id,
            optimizer.name,
            str(sorted(dataclasses.asdict(algorithm_cfg).items())),
            str(sorted(dataclasses.asdict(optimizer.cfg).items())),
        ]
    )
    digest = hashlib.md5(payload.encode()).hexdigest()[:8]
    return f"{algorithm}__{env_id}__{optimizer.name}__{digest}"


class WandbLogger:
    def __init__(
        self,
        entity=None,
        project=None,
        name=None,
        group=None,
        mode="disabled",
        cfg=None,
        seed=0,
        num_seeds=1,
        **kwargs,
    ):
        cfg = cfg or {}
        self.runs = {
            i: wandb.init(
                entity=entity,
                project=project,
                name=name,
                group=group,
                mode=mode,
                config={
                    **cfg,
                    "optimizer_config": config_str(cfg),
                    "seed": seed + i,
                },
                reinit="create_new",
            )
            for i in range(num_seeds)
        }

    def log(self, data: PyTree, step: int, **kwargs) -> None:
        num_seeds = len(self.runs)
        data = {
            "/".join(str(p.key) for p in path): ensure_axis(leaf, num_seeds)
            for path, leaf in jax.tree_util.tree_leaves_with_path(data)
        }
        for seed, run in self.runs.items():
            run.log({k: v[seed].mean() for k, v in data.items()}, step=step)

    def finish(self) -> None:
        for run in self.runs.values():
            run.finish()
