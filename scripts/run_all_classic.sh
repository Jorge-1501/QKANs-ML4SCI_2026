#!/usr/bin/env bash
# Classical KAN -> QKAN -> metrics collection for one seed. Extra arguments
# (e.g. --full-dataset) are forwarded to both training scripts.
    python3 scripts/train_kan.py --seed 42 --force "$@"
    python3 scripts/train_qkan.py --seed 42 --force "$@"
    python3 scripts/collect_metrics.py --task top
