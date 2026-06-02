import argparse

import numpy as np

from wandb_client import fetch_wandb


def interpolate(run_df, return_key, checkpoints):
    run_df = run_df.dropna(subset=[return_key]).sort_values("_step")
    if run_df.empty:
        return np.full(len(checkpoints), np.nan)
    return np.interp(
        checkpoints,
        run_df["_step"].to_numpy(),
        run_df[return_key].to_numpy(),
        left=np.nan,
        right=run_df[return_key].to_numpy()[-1],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--entity", default="noahfarr")
    p.add_argument("--project", default="stremax")
    p.add_argument("--return-key", default="training/episode_returns")
    p.add_argument("--checkpoints", type=int, default=6)
    p.add_argument("--clear-cache", action="store_true")
    args = p.parse_args()

    if args.clear_cache:
        from wandb_client import memory

        memory.clear(warn=False)

    df = fetch_wandb(
        args.entity,
        args.project,
        keys=[args.return_key],
        filters={"config.env_id": {"$regex": "MinAtar"}},
    )

    for col in ["env_id", "algorithm", "optimizer", args.return_key, "_step", "run_id"]:
        if col not in df.columns:
            raise SystemExit(f"missing column {col!r}; have {sorted(df.columns)}")

    def truthy(value):
        return str(value).lower() == "true"

    def variant(row):
        name = row["optimizer"]
        if name == "obgd":
            return "obgd-exact" if truthy(row.get("optimizer/exact")) else "obgd"
        if name != "calibrated":
            return name
        return f"calibrated[nu={row.get('optimizer/nu')}]"

    df["variant"] = df.apply(variant, axis=1)

    for env_id in sorted(df["env_id"].dropna().unique()):
        env = df[df["env_id"] == env_id]
        max_step = env["_step"].max()
        checkpoints = np.linspace(max_step / args.checkpoints, max_step, args.checkpoints)
        labels = [f"{c/1e6:.1f}M" for c in checkpoints]

        print(f"\n=== {env_id}  (episode return, mean±std over seeds; up to {max_step/1e6:.1f}M) ===\n")
        header = f"{'algorithm / variant':<30}{'seeds':>6}" + "".join(f"{l:>13}" for l in labels)
        print(header)
        print("-" * len(header))

        rows = []
        for (algo, opt), grp in env.groupby(["algorithm", "variant"], dropna=False):
            curves = [
                interpolate(grp[grp["run_id"] == rid], args.return_key, checkpoints)
                for rid in grp["run_id"].unique()
            ]
            curves = np.array(curves)
            mean = np.nanmean(curves, axis=0)
            std = np.nanstd(curves, axis=0)
            rows.append((f"{algo} / {opt}", len(curves), mean, std, mean[-1]))

        for label, n, mean, std, _ in sorted(rows, key=lambda r: -np.nan_to_num(r[4])):
            cells = "".join(
                f"{m:>6.1f}±{s:<5.1f}" if np.isfinite(m) else f"{'-':>13}"
                for m, s in zip(mean, std)
            )
            print(f"{label:<30}{n:>6}{cells}")


if __name__ == "__main__":
    main()
