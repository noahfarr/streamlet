#!/usr/bin/env bash
# Run the Q-learning MinAtar examples (stream-Q, implicit-Q, intentional-Q) with wandb logging.
set -euo pipefail

cd "$(dirname "$0")/.."

for example in stream_q_minatar implicit_q_minatar intentional_q_minatar; do
    echo "=== Running ${example} ==="
    uv run python "examples/${example}.py" --wandb
done
