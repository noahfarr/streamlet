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
        """Replay per-timestep sequences into each seed's run.

        Leaves may have *different* lengths along their time axis: each leaf is
        treated as a uniform sequence spanning the same global-step interval
        ``[steps[0], steps[-1]]`` and its points are placed evenly across it, so
        env-rate episode metrics and update-rate loss metrics coexist without a
        shared grid. Trailing ``*rest`` axes are reduced with ``nanmean``;
        ``NaN`` marks "no datapoint" and is skipped. Points from different leaves
        landing on the same global step are merged into one row. wandb batches
        the uploads internally.
        """
        steps = np.asarray(jax.device_get(steps)).reshape(-1)
        start, span = int(steps[0]), int(steps[-1]) - int(steps[0])
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
        import os
        import tempfile

        import orbax.checkpoint as ocp

        checkpointer = ocp.StandardCheckpointer()
        for seed, run in self.runs.items():
            seed_state = jax.tree.map(lambda x: x[seed], state)
            with tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "model")
                checkpointer.save(path, seed_state)
                checkpointer.wait_until_finished()
                artifact = wandb.Artifact(f"model-{run.id}", type="model")
                artifact.add_dir(path, name="model")
                run.log_artifact(artifact)

    def finish(self) -> None:
        for run in self.runs.values():
            run.finish()
