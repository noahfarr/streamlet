#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from slurmpilot import JobCreationInfo, SlurmPilot, unify

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_NAME = REPO_ROOT.name

MINATAR_ENV_IDS = [
    "gymnax::Breakout-MinAtar",
    "gymnax::Asterix-MinAtar",
    "gymnax::Freeway-MinAtar",
    "gymnax::SpaceInvaders-MinAtar",
]

DEVICE_PRESETS = {
    "cpu": dict(
        n_cpus=4,
        n_gpus=0,
        mem=8000,
        max_runtime_minutes=6 * 60,
        env={
            "JAX_PLATFORMS": "cpu",
            "PYTHONUNBUFFERED": "1",
            "VIRTUAL_ENV": "~/stremax/.venv",
        },
    ),
    "gpu": dict(
        n_cpus=4,
        n_gpus=1,
        mem=8000,
        max_runtime_minutes=6 * 60,
        env={
            "JAX_PLATFORMS": "cuda,cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
            "PYTHONUNBUFFERED": "1",
            "VIRTUAL_ENV": "~/stremax/.venv",
        },
    ),
}

DEFAULT_SETUP = (
    f'export PATH="$HOME/.local/bin:$PATH"'
)
PYTHON_BINARY = "uv run --no-sync python"


def build_sweep(
    env_ids: list[str],
    wandb: bool,
) -> list[str]:
    suffix = " --wandb" if wandb else ""

    return [f"--env-id={env_id}{suffix}" for env_id in env_ids]


def submit(
    slurm: SlurmPilot,
    *,
    example: str,
    device: str = "cpu",
    cluster: str = "ias",
    partition: str = "stud",
    env_ids: list[str] | None = None,
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
        env_ids=env_ids if env_ids is not None else MINATAR_ENV_IDS,
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
        sbatch_arguments="--output=logs/stdout-%a --error=logs/stderr-%a --exclude=dgx-station",
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
    p.add_argument(
        "--example", default="measured_q_minatar", help="examples/<name>.py to run."
    )
    p.add_argument("--env-ids", nargs="+", default=MINATAR_ENV_IDS)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--cluster", default="ias", help="slurmpilot cluster name.")
    p.add_argument("--partition", default="gpu")
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
