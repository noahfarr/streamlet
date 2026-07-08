import dataclasses
import hashlib
import warnings

import jax
import numpy as np
import wandb

from streamlet.utils.typing import PyTree


def config_str(cfg, prefix="optimizer/"):
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
        steps = np.asarray(jax.device_get(steps)).reshape(-1)
        start, span = int(steps[0]), int(steps[-1]) - int(steps[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            data = {
                "/".join(str(p.key) for p in path): np.nanmean(
                    np.asarray(jax.device_get(leaf)),
                    axis=tuple(range(2, np.ndim(leaf))),
                )
                for path, leaf in jax.tree_util.tree_leaves_with_path(data)
            }
        for seed, run in self.runs.items():
            rows: dict[int, dict] = {}
            for k, v in data.items():
                length = v.shape[1]
                grid = start + (np.arange(length) + 1) * span // length
                for t, step in enumerate(grid):
                    if np.isfinite(v[seed, t]):
                        rows.setdefault(int(step), {})[k] = float(v[seed, t])
            for step in sorted(rows):
                run.log(rows[step], step=step)

    def log_artifact(self, state: PyTree, step: int, **kwargs) -> None:
        from flax import serialization

        for seed, run in self.runs.items():
            seed_state = jax.tree.map(lambda x: x[seed], state)
            artifact = wandb.Artifact(f"model-{run.id}", type="model")
            with artifact.new_file("model.msgpack", mode="wb") as f:
                f.write(serialization.to_bytes(seed_state))
            run.log_artifact(artifact)

    def finish(self) -> None:
        for run in self.runs.values():
            run.finish()
