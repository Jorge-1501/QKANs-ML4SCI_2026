#!/usr/bin/env bash
# Full top-tagging pipeline (classical KAN -> QKAN), one run per seed.
#
# Usage: scripts/run_seeds.sh [--full-dataset]
#   (default)        hyperparams.py's regime: invariant-mass cut + n_subsets disjoint
#                    partitions; seed % n_subsets selects the subset (SEEDS default: 13 14)
#   --full-dataset   entire dataset: no mass cut, n_subsets=1, train/val/test kept separate
#                    (SEEDS default: 42)
# Override the seeds with e.g. SEEDS="10 11 12" scripts/run_seeds.sh.
#
# The regime is enforced end to end: the same flag is passed to preprocessing and to every
# training step, and each regime writes to its own directories
# (see src/utils/workspace.py: outputs/top/<cut>/<full|n{N}_subset{k}>/seed_<seed>/).
# Self-backgrounds via nohup on first invocation so it survives terminal/session exit.
set -e

SCRIPT_PATH="$(readlink -f "$0")"

cd "$(dirname "$SCRIPT_PATH")/.."

FULL_ARGS=()
REGIME="partitioned"
DEFAULT_SEEDS="13 14"
if [ "$1" = "--full-dataset" ]; then
    FULL_ARGS=(--full-dataset)
    REGIME="full"
    DEFAULT_SEEDS="42"
elif [ -n "$1" ]; then
    echo "Unknown argument: $1 (usage: $0 [--full-dataset])" >&2
    exit 2
fi
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"

LOG_DIR="$(PYTHONPATH=. python3 -c 'from src.utils.workspace import get_config; print(get_config("top", 0)["pipeline_logs_dir"])')"
mkdir -p "$LOG_DIR"
PIPELINE_LOG="${LOG_DIR}/pipeline_${REGIME}_$(date +%Y%m%d_%H%M%S).log"

if [ -z "$PIPELINE_BACKGROUNDED" ]; then
    export PIPELINE_BACKGROUNDED=1
    nohup "$SCRIPT_PATH" "$@" > "$PIPELINE_LOG" 2>&1 &
    echo "Pipeline launched in background (PID $!)."
    echo "Combined log: $PIPELINE_LOG"
    echo "Per-seed logs are written under the run directory's logs/ (see workspace.get_config)."
    exit 0
fi

echo "[1/3] Preprocessing top-tagging data (regime: ${REGIME})..."
# --force rebuilds the canonical partition for THIS regime only (each regime has its own
# cache directory); training scripts only ever select from a prebuilt partition.
if [ "$REGIME" = "full" ]; then
    python3 scripts/run_preprocessing.py --full-dataset --force
else
    python3 scripts/run_preprocessing.py --force
fi

echo "[2/3] Training per seed: ${SEEDS}"
for SEED in $SEEDS; do
    echo "========================================================================"
    echo "STARTING PIPELINE RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"

    echo "[2a] Training classical KAN pipeline (seed=${SEED})..."
    python3 scripts/train_kan.py --seed ${SEED} "${FULL_ARGS[@]}"

    echo "[2b] Training QKAN pipeline (seed=${SEED})..."
    python3 scripts/train_qkan.py --seed ${SEED} --task top "${FULL_ARGS[@]}"

    echo "========================================================================"
    echo "FINISHED PIPELINE RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"
done

echo "[3/3] Refreshing the aggregate metrics table..."
python3 scripts/collect_metrics.py --task top
