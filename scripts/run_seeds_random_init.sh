#!/usr/bin/env bash
# Brief ablation: retrain the VQC from random weights (instead of the
# KAN-extracted warm start) for seeds 10-14, ideal backend only. Reuses the
# classical KAN models / extracted quantum_weights.pt graphs already present in
# each seed's run directory -- does not touch classical training. Extra arguments
# (e.g. --full-dataset) are forwarded to train_qkan.py and must match the regime
# the classical models were trained under.
set -e

cd "$(dirname "$(readlink -f "$0")")/.."

for SEED in 10 11 12 13 14; do
    echo "========================================================================"
    echo "RANDOM-INIT VQC RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"
    python3 scripts/train_qkan.py --seed ${SEED} --task top --random_init "$@"
done
