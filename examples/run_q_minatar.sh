#!/usr/bin/env bash
# Run the Q-learning MinAtar examples (stream-Q, implicit-Q, intentional-Q) across all MinAtar envs with wandb logging.
set -euo pipefail

cd "$(dirname "$0")/.."

envs=(
    "gymnax::Breakout-MinAtar"
    "gymnax::Asterix-MinAtar"
    "gymnax::Freeway-MinAtar"
    "gymnax::SpaceInvaders-MinAtar"
)

implicit_lrs=(0.001)

for example in measured_q_minatar; do
    for env_id in "${envs[@]}"; do
        echo "=== Running ${example} on ${env_id} ==="
        uv run python "examples/${example}.py" --env-id "${env_id}" --wandb
    done
done
