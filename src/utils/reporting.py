# src/utils/reporting.py
# Collects the per-run/per-stage metrics JSON files that classic_kan.py /
# quantum_kan.py already write (unchanged) into one long-format table, tagged by
# task/seed/variant/subset_id/model, for later Parquet export and cross-run tabular
# extraction/plotting. Pure collection -- no statistics (mean/std, ...) computed
# here; that analysis is deliberately left for later, separate work.
import json
import os

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
    "qkan_random_ideal": "metrics_qkan_random_ideal",
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
    """Walks every run directory actually present under outputs/<task>/ (see
    workspace.iter_run_dirs), and for each one, for every model/stage in
    METRIC_REGISTRY, collects that stage's metrics JSON (if it exists) into one row
    tagged with task/seed/variant/apply_mass_cut/n_subsets/subset_id/model.

    Despite the name, this is a COLLECTION function, not a statistical one: it
    does not compute mean/std or any other aggregate -- it only gathers and tags
    whatever per-run metrics already exist on disk into a single long-format
    DataFrame (one row per (run, model) pair found), ready for Parquet export and
    later analysis elsewhere. Missing seeds/stages (partial sweeps) are silently
    skipped, not errors.

    The data regime tags come from each run's directory name (workspace's variant
    layout), not from the current hyperparams.py, so a full-dataset run is never
    mislabeled as one of n_subsets partitions. Pre-variant-layout runs are tagged
    variant="legacy" with apply_mass_cut/n_subsets/subset_id left empty.
    """
    rows = []
    for run in workspace.iter_run_dirs(task):
        seed = run["seed"]
        if run["legacy"]:
            config = workspace.get_config(task, seed)
        else:
            config = workspace.get_config(
                task, seed, apply_mass_cut=run["apply_mass_cut"], n_subsets=run["n_subsets"]
            )

        tags = {
            "task": task,
            "seed": seed,
            "variant": run["variant"],
            "apply_mass_cut": run["apply_mass_cut"],
            "n_subsets": run["n_subsets"],
            "subset_id": run["subset_id"],
        }

        for model_name, config_key in METRIC_REGISTRY.items():
            path = config.get(config_key)
            if path:
                # get_config's paths are rooted at the variant it was built for; re-root
                # onto the directory actually found on disk (identical except for legacy runs).
                path = os.path.join(run["path"], os.path.relpath(path, config["run_dir"]))
            row = _flatten_metrics_file(path, tags={**tags, "model": model_name})
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)
