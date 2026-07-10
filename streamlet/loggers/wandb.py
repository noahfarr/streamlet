import dataclasses
import hashlib

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
        start, end = int(steps.min()), int(steps.max())
        span = end - start
        data = jax.device_get(data)

        rows: list[dict[int, dict]] = [{} for _ in self.runs]
        for k, v in data.items():
            for seed, row in enumerate(v):
                row = np.asarray(row)
                length = len(row)
                if length == 0:
                    continue
                grid = start + (np.arange(length) + 1) * span // length
                finite = np.isfinite(row)
                for step, value in zip(grid[finite].tolist(), row[finite].tolist()):
                    rows[seed].setdefault(step, {})[k] = value

        for seed, run in self.runs.items():
            for step in sorted(rows[seed]):
                run.log(rows[seed][step], step=step)

    def log_summary(self, data: PyTree, **kwargs) -> None:
        data = jax.device_get(data)
        for seed, run in self.runs.items():
            for k, v in data.items():
                value = np.asarray(v)[seed]
                if np.isfinite(value):
                    run.summary[k] = float(value)

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
