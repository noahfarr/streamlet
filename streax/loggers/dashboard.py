from collections import defaultdict
from typing import Any

import jax
import jax.numpy as jnp
from rich import box
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from streax.utils.typing import PyTree


class DashboardLogger:
    def __init__(
        self,
        total_timesteps=0,
        refresh_per_second=10,
        summary=None,
        title="Streax",
        **kwargs,
    ):
        self.summary = summary or {}
        self.title = title

        self.console = Console()

        self.progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            SpinnerColumn(),
            TimeElapsedColumn(),
            BarColumn(bar_width=None),
            TimeRemainingColumn(),
            expand=True,
            console=self.console,
        )
        self.progress_task = self.progress.add_task("Progress", total=total_timesteps)

        dashboard = self.build_dashboard({}, 0, self.progress, self.progress_task)

        self.live = Live(
            dashboard,
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=False,
        )
        self.live.start()

    def log(self, data: PyTree, steps: PyTree, **kwargs) -> None:
        """Render a snapshot of a per-timestep sequence.

        Leaves are ``(num_seeds, T, *rest)``; the trailing ``*rest`` axes are
        reduced with ``nanmean`` and the most recent finite value along ``T`` is
        kept per seed, giving a ``(num_seeds,)`` snapshot per key (the table then
        shows ``mean ± std`` across seeds). ``NaN``-only keys are dropped.
        """
        steps = jnp.asarray(steps).reshape(-1)
        step = int(steps[-1])
        snapshot = {
            "/".join(str(p.key) for p in path): self._last_finite(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(data)
        }
        snapshot = {k: v for k, v in snapshot.items() if v is not None}
        self.progress.update(self.progress_task, completed=step)
        dashboard = self.build_dashboard(
            snapshot, step, self.progress, self.progress_task
        )
        self.live.update(dashboard, refresh=True)

    @staticmethod
    def _last_finite(leaf: PyTree):
        """Reduce a ``(num_seeds, T, *rest)`` leaf to the most recent finite
        per-seed value, or ``None`` if no finite value exists."""
        leaf = jnp.asarray(leaf)
        reduced = jnp.nanmean(leaf, axis=tuple(range(2, leaf.ndim)))  # (S, T)
        finite = jnp.isfinite(reduced)
        if not bool(jnp.any(finite)):
            return None
        # index of the last finite step per seed (0 if a seed has none)
        last = (reduced.shape[1] - 1) - jnp.argmax(finite[:, ::-1], axis=1)
        return jnp.take_along_axis(reduced, last[:, None], axis=1)[:, 0]

    def finish(self) -> None:
        self.live.stop()
        self.console.show_cursor(True)

    def group(self, data: dict[str, PyTree]) -> dict[str, dict[str, Any]]:
        data = {
            "/".join(str(p.key) for p in path): leaf
            for path, leaf in jax.tree_util.tree_leaves_with_path(data)
        }
        groups = defaultdict(dict)
        for key, value in data.items():
            if "/" in key:
                prefix, name = key.split("/", 1)
                groups[prefix][name] = value
            else:
                groups[""][key] = value
        return dict(groups)

    def build_table(self, heading: str, metrics: dict[str, PyTree]) -> Table:
        table = Table(box=None, expand=True)
        table.add_column(heading, justify="left", width=20, style="yellow")
        table.add_column("Value", justify="right", width=10, style="green")
        for name, value in metrics.items():
            mean, std = jnp.mean(value), jnp.std(value)
            if 0 < abs(mean) < 0.001:
                fmt = ".3e"
            elif abs(mean) >= 10000:
                fmt = "_.0f"
            else:
                fmt = ".3f"
            value_str = f"{mean:{fmt}} ± {std:{fmt}}" if std != 0 else f"{mean:{fmt}}"
            table.add_row(name, value_str)
        return table

    def build_dashboard(
        self, data: dict[str, PyTree], step: int, progress: Progress, task: Any
    ) -> Table:
        dashboard = Table(
            box=box.ROUNDED,
            expand=True,
            show_header=False,
            border_style="white",
            title=self.title,
            title_style="bold",
        )

        dynamic_summary = {
            k.split("/", 1)[1]: v for k, v in data.items() if k.startswith("summary/")
        }
        items = [*self.summary.items(), *dynamic_summary.items()]
        if data:
            items.append(("Step", f"{int(step):_}"))
        left = Table(box=None, expand=True)
        left.add_column("Summary", justify="left", width=16, style="white")
        left.add_column("Value", justify="right", width=8, style="white")
        right = Table(box=None, expand=True)
        right.add_column("Summary", justify="left", width=16, style="white")
        right.add_column("Value", justify="right", width=8, style="white")
        for i, (key, value) in enumerate(items):
            table = left if i % 2 == 0 else right
            value_str = f"{value:_}" if isinstance(value, int) else f"{value}"
            table.add_row(key, value_str, style="white")
        summary_row = Table(box=None, expand=True, pad_edge=False)
        summary_row.add_row(left, right)
        dashboard.add_row(summary_row)

        groups = self.group(data)
        groups.pop("summary", None)
        group_names = list(groups.keys())

        for i in range(0, len(group_names), 2):
            pair = group_names[i : i + 2]
            tables = [self.build_table(name, groups[name]) for name in pair]
            row = Table(box=None, expand=True, pad_edge=False)
            row.add_row(*tables)
            dashboard.add_row(row)

        dashboard.add_row("")
        progress.update(task, completed=int(step))
        dashboard.add_row(progress)

        return dashboard
