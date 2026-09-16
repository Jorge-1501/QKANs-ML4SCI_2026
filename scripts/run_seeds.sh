#!/usr/bin/env bash
# Full top-tagging pipeline (classical KAN -> QKAN) run for seeds 10 to 14
# under hyperparams.py's n_subsets (disjoint partitions/statistical replicates) and
# apply_mass_cut=False (no invariant-mass cut) defaults. Self-backgrounds via
# nohup on first invocation so it survives terminal/session exit.
set -e

SCRIPT_PATH="$(readlink -f "$0")"

cd "$(dirname "$SCRIPT_PATH")/.."

LOG_DIR="outputs/top"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="${LOG_DIR}/pipeline_no_split_no_masscut_$(date +%Y%m%d_%H%M%S).log"

if [ -z "$PIPELINE_BACKGROUNDED" ]; then
    export PIPELINE_BACKGROUNDED=1
    nohup "$SCRIPT_PATH" > "$PIPELINE_LOG" 2>&1 &
    echo "Pipeline launched in background (PID $!)."
    echo "Combined log: $PIPELINE_LOG"
    echo "Per-seed particular logs will be written under outputs/top/seed_<seed>/logs/"
    exit 0
fi

#echo "[1/2] Preprocessing top-tagging data (split and mass cut)..."
# We run preprocessing with --force once to ensure the 'no_mass_cut' canonical
# partition is fully built if it doesn't exist yet, as training scripts only select
# from prebuilt partitions.
#python3 scripts/run_preprocessing.py  --force

for SEED in {13..14}; do
    echo "========================================================================"
    echo "STARTING PIPELINE RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"
    
    echo "[2a] Training classical KAN pipeline (seed=${SEED})..."
    python3 scripts/train_kan.py --seed ${SEED}

    echo "[2b] Training QKAN pipeline (seed=${SEED})..."
    python3 scripts/train_qkan.py --seed ${SEED} --task top
    
    echo "========================================================================"
    echo "FINISHED PIPELINE RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"
done

