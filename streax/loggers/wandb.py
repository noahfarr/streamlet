import dataclasses
import hashlib
import warnings

import jax
import numpy as np
import wandb

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

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        """Replay a per-timestep sequence into each seed's run.

        Every leaf is expected to have shape ``(num_seeds, T, *rest)``; the
        trailing ``*rest`` axes (e.g. the env axis) are reduced with ``nanmean``
        to ``(num_seeds, T)``. ``steps`` is a length-``T`` array of absolute
        global step indices. ``NaN`` marks "no datapoint" for a key at a given
        ``(seed, step)`` and is skipped, so sparse episode metrics and dense
        loss metrics can share one grid. wandb batches the uploads internally.
        """
        steps = np.asarray(jax.device_get(steps)).reshape(-1)
        with warnings.catch_warnings():
            # all-NaN env slices (steps with no finished episode) -> NaN, skipped below
            warnings.simplefilter("ignore", RuntimeWarning)
            data = {
                "/".join(str(p.key) for p in path): np.nanmean(
                    np.asarray(jax.device_get(leaf)),
                    axis=tuple(range(2, np.ndim(leaf))),
                )
                for path, leaf in jax.tree_util.tree_leaves_with_path(data)
            }
        for seed, run in self.runs.items():
            for t, step in enumerate(steps):
                row = {
                    k: float(v[seed, t])
                    for k, v in data.items()
                    if np.isfinite(v[seed, t])
                }
                if row:
                    run.log(row, step=int(step))

    def finish(self) -> None:
        for run in self.runs.values():
            run.finish()
