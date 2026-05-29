#!/usr/bin/env bash
# Submit stremax MinAtar examples to SLURM. Run on the ias login node from the repo root.
# Defaults to the VOGD eta sweep across MinAtar envs; override via env vars (EXAMPLE, ETAS, ENVS, PARTITION, ...).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

EXAMPLE="${EXAMPLE:-vogd_q_minatar}"
PARTITION="${PARTITION:-stud}"
GRES="${GRES:-gpu:1}"
CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"
TIME="${TIME:-08:00:00}"
WANDB="${WANDB:-0}"

read -r -a ENVS <<< "${ENVS:-gymnax::Breakout-MinAtar gymnax::Asterix-MinAtar gymnax::Freeway-MinAtar gymnax::SpaceInvaders-MinAtar}"
read -r -a ETAS <<< "${ETAS:-0.25 0.5 1.0}"

cd "$(dirname "$0")/.."
mkdir -p logs
uv sync

wandb_flag=""
[ "$WANDB" = "1" ] && wandb_flag="--wandb"

submit() {
    name="$1"
    shift
    sbatch --job-name="$name" \
        --partition="$PARTITION" --gres="$GRES" \
        --cpus-per-task="$CPUS" --mem="$MEM" --time="$TIME" \
        --output="logs/%x-%j.out" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH="\$HOME/.local/bin:\$PATH"
echo "host=\$(hostname) job=\$SLURM_JOB_ID gpu=\${CUDA_VISIBLE_DEVICES:-none}"
command -v nvidia-smi >/dev/null && nvidia-smi -L || true
uv run --no-sync python "examples/${EXAMPLE}.py" $* ${wandb_flag}
EOF
}

for env_id in "${ENVS[@]}"; do
    short="${env_id#gymnax::}"
    short="${short%-MinAtar}"
    for eta in "${ETAS[@]}"; do
        submit "${EXAMPLE}-${short}-eta${eta}" --env-id "$env_id" --eta "$eta"
    done
done
