import os

import orbax.checkpoint as ocp

from streamlet.utils.typing import PyTree


class CheckpointLogger:
    def __init__(
        self,
        directory: str | None = None,
        max_to_keep: int = 1,
        best_mode: str = "max",
        cfg=None,
        **kwargs,
    ):
        if directory is None:
            out_dir = (cfg or {}).get("resume")
            if out_dir is None:
                try:
                    from hydra.core.hydra_config import HydraConfig

                    out_dir = HydraConfig.get().runtime.output_dir
                except Exception:
                    out_dir = "."
            directory = os.path.join(out_dir, "checkpoints")
        self.latest = ocp.CheckpointManager(
            os.path.join(directory, "latest"),
            options=ocp.CheckpointManagerOptions(max_to_keep=max_to_keep),
        )
        self.best = ocp.CheckpointManager(
            os.path.join(directory, "best"),
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                best_fn=lambda metrics: metrics["mean_return"],
                best_mode=best_mode,
            ),
        )

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        pass

    def log_summary(self, data: PyTree, **kwargs) -> None:
        pass

    def log_artifact(self, state: PyTree, step: int, metrics=None, **kwargs) -> None:
        self.latest.save(step, args=ocp.args.StandardSave(state))
        self.best.save(step, args=ocp.args.StandardSave(state), metrics=metrics or {})

    def restore(self, state: PyTree) -> tuple[PyTree, int]:
        step = self.latest.latest_step()
        if step is None:
            return state, 0
        state = self.latest.restore(step, args=ocp.args.StandardRestore(state))
        return state, step + 1

    def finish(self) -> None:
        self.latest.wait_until_finished()
        self.best.wait_until_finished()
