#!/usr/bin/env bash
# Brief ablation: retrain the VQC from random weights (instead of the
# KAN-extracted warm start) for seeds 10-14, ideal backend only. Reuses the
# classical KAN models / extracted quantum_weights.pt graphs already present
# under outputs/top/seed_{10..14} -- does not touch classical training.
set -e

cd "$(dirname "$(readlink -f "$0")")/.."

for SEED in 10 11 12 13 14; do
    echo "========================================================================"
    echo "RANDOM-INIT VQC RUN FOR SEED: ${SEED} ($(date))"
    echo "========================================================================"
    python3 scripts/train_qkan.py --seed ${SEED} --task top --random_init
done
