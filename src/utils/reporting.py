# src/utils/reporting.py
# Collects the per-run/per-stage metrics JSON files that classic_kan.py /
# quantum_kan.py already write (unchanged) into one long-format table, tagged by
# task/seed/subset_id/model, for later Parquet export and cross-run tabular
# extraction/plotting. Pure collection -- no statistics (mean/std, ...) computed
# here; that analysis is deliberately left for later, separate work.
import json
import os
from pathlib import Path

import pandas as pd

from src.utils import workspace

# model/stage name -> the get_config() key holding that stage's metrics JSON path.
# Extend this registry (not compute_run_statistics itself) to add a future
# architecture's metrics to the collected table.
METRIC_REGISTRY = {
    "classical_base": "base_eval_metrics",
    "classical_retrained": "retrain_eval_metrics",
    "classical_symbolic": "symbolic_eval_metrics",
    "classical_final": "final_eval_metrics",
    "qkan_ideal": "metrics_qkan_ideal",
    "qkan_noisy": "metrics_qkan_noisy",
    "qkan_shots": "metrics_qkan_shots",
    "qkan_baseline_ideal": "metrics_qkan_baseline_ideal",
    "qkan_baseline_noisy": "metrics_qkan_baseline_noisy",
    "qkan_baseline_shots": "metrics_qkan_baseline_shots",
    "random_forest": "rf_eval_metrics",
}


def _flatten_metrics_file(path, tags):
    """Reads one metrics JSON file if it exists and merges in the given tag
    columns (e.g. task/seed/subset_id/model). List/dict values (e.g. "Confusion
    Matrix") are JSON-serialized to strings so every cell stays a scalar. Returns
    None if the file doesn't exist -- pure pass-through, no computation."""
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        metrics = json.load(f)

    row = dict(tags)
    for key, value in metrics.items():
        row[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
    return row


def compute_run_statistics(task):
    """Globs every outputs/<task>/seed_*/ directory actually present on disk, and
    for each one, for every model/stage in METRIC_REGISTRY, collects that stage's
    metrics JSON (if it exists) into one row tagged with task/seed/subset_id/model.

    Despite the name, this is a COLLECTION function, not a statistical one: it
    does not compute mean/std or any other aggregate -- it only gathers and tags
    whatever per-run metrics already exist on disk into a single long-format
    DataFrame (one row per (seed, model) pair found), ready for Parquet export and
    later analysis elsewhere. Missing seeds/stages (partial sweeps) are silently
    skipped, not errors.
    """
    root = workspace.get_project_root()
    outputs_task_dir = Path(root) / "outputs" / task
    seed_dirs = sorted(outputs_task_dir.glob("seed_*"))

    rows = []
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        config = workspace.get_config(task, seed)
        n_subsets = config.get("n_subsets", 15)
        subset_id = seed % n_subsets

        for model_name, config_key in METRIC_REGISTRY.items():
            path = config.get(config_key)
            row = _flatten_metrics_file(
                path,
                tags={"task": task, "seed": seed, "subset_id": subset_id, "model": model_name},
            )
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)
