#!/usr/bin/env bash
# Full top-tagging pipeline (preprocess -> classical KAN -> QKAN) run under
# hyperparams.py's n_subsets=1 (no statistical-replicate split) and
# apply_mass_cut=False (no invariant-mass cut) defaults. Self-backgrounds via
# nohup on first invocation so it survives terminal/session exit.
set -e

SCRIPT_PATH="$(readlink -f "$0")"

cd "$(dirname "$SCRIPT_PATH")/.."

SEED=39
LOG_DIR="outputs/top/seed_${SEED}/logs"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="${LOG_DIR}/pipeline_no_split_no_masscut_$(date +%Y%m%d_%H%M%S).log"

if [ -z "$PIPELINE_BACKGROUNDED" ]; then
    export PIPELINE_BACKGROUNDED=1
    nohup "$SCRIPT_PATH" > "$PIPELINE_LOG" 2>&1 &
    echo "Pipeline launched in background (PID $!)."
    echo "Combined log: $PIPELINE_LOG"
    echo "Per-stage logs also written under $LOG_DIR"
    exit 0
fi

echo "[1/3] Preprocessing top-tagging data (no split, no mass cut)..."
python3 scripts/run_preprocessing.py --no-apply-mass-cut

echo "[2/3] Training classical KAN pipeline (seed=${SEED})..."
python3 scripts/train_kan.py --seed ${SEED} --force

echo "[3/3] Training QKAN pipeline (seed=${SEED})..."
python3 scripts/train_qkan.py --seed ${SEED} --force --task top
