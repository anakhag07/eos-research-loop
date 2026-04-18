#!/usr/bin/env bash
# run-once.sh <project-dir>
#
# Stage-0 non-interactive pipeline: produce a per-run CSV summary and a
# timestamped directory of PNG plots for <project-dir>, without requiring
# a display or any user input.
#
# Usage:
#   bash run-once.sh eoss-resnet-baselines
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <project-dir>" >&2
    exit 2
fi

PROJECT_DIR="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

if [[ ! -d "$PROJECT_DIR" ]]; then
    echo "project dir not found: $PROJECT_DIR" >&2
    exit 2
fi

CONDA_SH="/orcd/software/core/001/pkg/miniforge/24.3.0-0/etc/profile.d/conda.sh"
if [[ -f "$CONDA_SH" ]]; then
    # shellcheck disable=SC1090
    source "$CONDA_SH"
    conda activate eoss
fi

export MPLBACKEND=Agg
unset DISPLAY || true

CSV_OUT="$PROJECT_DIR/runs.csv"
echo "[run-once] summarizing -> $CSV_OUT"
python eoss_training_scripts/summarize_wandb_runs.py \
    --wandb-root "$PROJECT_DIR" \
    --output "$CSV_OUT"

echo "[run-once] plotting (headless)"
python generate_plots_wandb.py --headless

echo "[run-once] done."
