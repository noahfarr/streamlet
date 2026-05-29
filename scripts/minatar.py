#!/usr/bin/env python
from __future__ import annotations

import argparse

from launch import DEFAULT_SETUP, MINATAR_ENV_IDS, SlurmPilot, submit

OPTIMIZER_EXAMPLES = {
    "stream_q": "stream_q_minatar",
    "measured_q": "measured_q_minatar",
    "implicit_q": "implicit_q_minatar",
    "intentional_q": "intentional_q_minatar",
    "adaptive_q": "adaptive_q_minatar",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--optimizers",
        nargs="+",
        choices=list(OPTIMIZER_EXAMPLES),
        default=list(OPTIMIZER_EXAMPLES),
        help="Which optimizers to launch (default: all).",
    )
    p.add_argument("--env-ids", nargs="+", default=MINATAR_ENV_IDS)
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--cluster", default="ias")
    p.add_argument("--partition", default="stud")
    p.add_argument("--wandb", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--setup", default=DEFAULT_SETUP)
    p.add_argument("--n-concurrent-jobs", type=int, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare the job files but do not submit to SLURM.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    slurm = SlurmPilot(clusters=[args.cluster])

    examples = [OPTIMIZER_EXAMPLES[name] for name in args.optimizers]
    print(f"Launching {len(examples)} optimizer(s) on MinAtar ({args.device}):\n")

    submitted = []
    for example in examples:
        jobname, _ = submit(
            slurm,
            example=example,
            device=args.device,
            cluster=args.cluster,
            partition=args.partition,
            env_ids=args.env_ids,
            wandb=args.wandb,
            setup=args.setup,
            n_concurrent_jobs=args.n_concurrent_jobs,
            dry_run=args.dry_run,
        )
        submitted.append(jobname)

    verb = "Prepared" if args.dry_run else "Submitted"
    print(f"{verb} {len(submitted)} array job(s):")
    for jobname in submitted:
        print(f"  {jobname}")


if __name__ == "__main__":
    main()
