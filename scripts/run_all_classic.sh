#!/usr/bin/env bash
# Classical KAN -> QKAN -> metrics collection for one seed. Extra arguments
# (e.g. --full-dataset) are forwarded to both training scripts.
for i in {11..14}; do
    python3 ../scripts/train_kan.py --seed ${i} --force 
    python3 ../scripts/train_qkan.py --seed ${i} --force
    python3 ../scripts/train_rf.py --seed ${i} --force --task top
done
python3 scripts/collect_metrics.py --task top