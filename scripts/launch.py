#!/usr/bin/env python
"""Launch stremax example scripts on a SLURM cluster via slurmpilot.

Replaces the old ``cluster/launch.sh`` heredoc. Submits an ``examples/*.py``
script as a SLURM *array* job, one task per point in the env-id (and, for
examples that take ``--eta``, eta) sweep. Run it from your laptop; slurmpilot
ships a copy of the repo to the cluster over SSH and calls ``sbatch`` there.

The upload is filtered by the repo's ``.gitignore`` (handled by slurmpilot's
``feat/spignore`` branch), so ``.venv/``, ``wandb/`` etc. are not shipped.

One-time setup (see scripts/slurmpilot/ for templates):

    mkdir -p ~/slurmpilot/config/clusters
    cp scripts/slurmpilot/general.yaml       ~/slurmpilot/config/general.yaml
    cp scripts/slurmpilot/clusters/ias.yaml  ~/slurmpilot/config/clusters/ias.yaml
    # then edit ~/slurmpilot/config/clusters/ias.yaml and set `host:` (and `user:`)

Install the launch tooling once (the fork is pulled from git):

    uv sync --extra cluster

Examples:

    # Default: Measured eta sweep across the 4 MinAtar envs on CPU, with wandb.
    uv run python scripts/launch.py

    # GPU, a different example, no eta sweep, no wandb:
    uv run python scripts/launch.py --example stream_q_minatar --device gpu --no-wandb

    # See exactly what would be submitted without touching SLURM:
    uv run python scripts/launch.py --dry-run

This module also exposes ``submit()`` so other scripts (e.g. minatar.py) can
launch several examples through one SlurmPilot connection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify

REPO_ROOT = Path(__file__).resolve().parent.parent
# Name of the repo dir as it lands on the cluster (job_dir/<SRC_NAME>/...).
SRC_NAME = REPO_ROOT.name

MINATAR_ENV_IDS = [
    "gymnax::Breakout-MinAtar",
    "gymnax::Asterix-MinAtar",
    "gymnax::Freeway-MinAtar",
    "gymnax::SpaceInvaders-MinAtar",
]
DEFAULT_ETAS = [0.25, 0.5, 1.0]
# Combinations to drop from the sweep, as "<env-id>:<eta>".
DEFAULT_SKIP = ["gymnax::Breakout-MinAtar:0.5"]

# Per-device resource + environment presets (mirrors the old launch.sh).
DEVICE_PRESETS = {
    "cpu": dict(
        n_cpus=8,
        n_gpus=0,
        mem=8000,
        max_runtime_minutes=24 * 60,
        env={"JAX_PLATFORMS": "cpu", "PYTHONUNBUFFERED": "1"},
    ),
    "gpu": dict(
        n_cpus=4,
        n_gpus=1,
        mem=16000,
        max_runtime_minutes=8 * 60,
        env={
            "JAX_PLATFORMS": "cuda,cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
            "PYTHONUNBUFFERED": "1",
        },
    ),
}

# Default node setup: put uv on PATH and build the env from the shipped uv.lock.
# Override with --setup if your compute nodes lack internet (e.g. point uv at a
# pre-built shared environment instead).
DEFAULT_SETUP = f'export PATH="$HOME/.local/bin:$PATH" && uv sync --frozen --project {SRC_NAME}'
# Run the example through the shipped project's environment. cwd on the node is
# the job dir, so the project lives in the <SRC_NAME>/ subdir and the entrypoint
# path slurmpilot generates ("<SRC_NAME>/examples/X.py") resolves from there.
PYTHON_BINARY = f"uv run --no-sync --project {SRC_NAME} python"


def example_takes_eta(example: str) -> bool:
    """True if examples/<example>.py defines a ``--eta`` argument."""
    source = (REPO_ROOT / "examples" / f"{example}.py").read_text()
    return "--eta" in source


def build_sweep(
    example: str,
    env_ids: list[str],
    etas: list[float],
    skip: set[str],
    wandb: bool,
) -> list[str]:
    """Build the list of per-task argument strings for the array job."""
    use_eta = example_takes_eta(example)
    suffix = " --wandb" if wandb else ""

    args: list[str] = []
    for env_id in env_ids:
        if use_eta:
            for eta in etas:
                if f"{env_id}:{eta}" in skip:
                    continue
                args.append(f"--env-id={env_id} --eta={eta}{suffix}")
        else:
            if env_id in skip:
                continue
            args.append(f"--env-id={env_id}{suffix}")
    return args


def submit(
    slurm: SlurmPilot,
    *,
    example: str,
    device: str = "cpu",
    cluster: str = "ias",
    partition: str = "stud",
    env_ids: list[str] | None = None,
    etas: list[float] | None = None,
    skip: set[str] | None = None,
    wandb: bool = True,
    setup: str = DEFAULT_SETUP,
    n_concurrent_jobs: int | None = None,
    n_cpus: int | None = None,
    n_gpus: int | None = None,
    mem: int | None = None,
    max_runtime_minutes: int | None = None,
    dry_run: bool = False,
) -> tuple[str, int | None]:
    """Submit one example as a SLURM array job. Returns (jobname, jobid)."""
    preset = DEVICE_PRESETS[device]
    sweep = build_sweep(
        example=example,
        env_ids=env_ids if env_ids is not None else MINATAR_ENV_IDS,
        etas=etas if etas is not None else DEFAULT_ETAS,
        skip=skip if skip is not None else set(DEFAULT_SKIP),
        wandb=wandb,
    )
    if not sweep:
        raise SystemExit(f"Sweep for {example} is empty (check env-ids/skip).")

    print(f"Submitting {len(sweep)} task(s) of examples/{example}.py:")
    for line in sweep:
        print(f"  {line}")

    job_info = JobCreationInfo(
        jobname=unify(f"{example}-{device}", method="coolname"),
        entrypoint=f"examples/{example}.py",
        cluster=cluster,
        src_dir=str(REPO_ROOT),
        python_binary=PYTHON_BINARY,
        python_args=sweep,
        n_concurrent_jobs=n_concurrent_jobs,
        bash_setup_command=setup,
        # Per-array-task log files; overrides slurmpilot's default logs/stdout
        # (without %a all tasks would clobber the same file).
        sbatch_arguments="--output=logs/stdout-%a --error=logs/stderr-%a",
        partition=partition,
        n_cpus=n_cpus or preset["n_cpus"],
        n_gpus=n_gpus if n_gpus is not None else preset["n_gpus"],
        mem=mem or preset["mem"],
        max_runtime_minutes=max_runtime_minutes or preset["max_runtime_minutes"],
        env=preset["env"],
    )
    jobid = slurm.schedule_job(job_info, dryrun=dry_run)

    if dry_run:
        print(f"  -> dry run: prepared '{job_info.jobname}' (not submitted)\n")
    else:
        print(f"  -> submitted '{job_info.jobname}' (SLURM id {jobid})")
        print(f"     sp status {job_info.jobname}   |   sp log {job_info.jobname}\n")
    return job_info.jobname, jobid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--example", default="measured_q_minatar", help="examples/<name>.py to run.")
    p.add_argument("--env-ids", nargs="+", default=MINATAR_ENV_IDS)
    p.add_argument(
        "--etas",
        nargs="*",
        type=float,
        default=DEFAULT_ETAS,
        help="Only used if the example defines --eta.",
    )
    p.add_argument(
        "--skip",
        nargs="*",
        default=DEFAULT_SKIP,
        help='Combinations to drop, as "<env-id>:<eta>" (or just "<env-id>").',
    )
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--cluster", default="ias", help="slurmpilot cluster name.")
    p.add_argument("--partition", default="stud")
    p.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass --wandb to the example (default: on).",
    )
    p.add_argument(
        "--setup",
        default=DEFAULT_SETUP,
        help="bash_setup_command run on the node before the job (env install).",
    )
    p.add_argument("--n-concurrent-jobs", type=int, default=None)
    # Per-run resource overrides (default to the device preset).
    p.add_argument("--n-cpus", type=int, default=None)
    p.add_argument("--n-gpus", type=int, default=None)
    p.add_argument("--mem", type=int, default=None, help="Memory in MB.")
    p.add_argument("--max-runtime-minutes", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the job files but do not submit to SLURM.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    slurm = SlurmPilot(clusters=[args.cluster])
    submit(
        slurm,
        example=args.example,
        device=args.device,
        cluster=args.cluster,
        partition=args.partition,
        env_ids=args.env_ids,
        etas=args.etas,
        skip=set(args.skip),
        wandb=args.wandb,
        setup=args.setup,
        n_concurrent_jobs=args.n_concurrent_jobs,
        n_cpus=args.n_cpus,
        n_gpus=args.n_gpus,
        mem=args.mem,
        max_runtime_minutes=args.max_runtime_minutes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
