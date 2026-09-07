# scripts/collect_metrics.py
# Collects the per-run/per-stage metrics JSON files already written across every
# outputs/<task>/seed_*/ directory into one long-format Parquet table, tagged by
# task/seed/subset_id/model.
import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
from src.utils.reporting import compute_run_statistics


def main(args):
    df = compute_run_statistics(args.task)

    config = workspace.get_config(args.task, seed=0)
    out_path = config["metrics_table_path"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)

    n_seeds = df["seed"].nunique() if len(df) else 0
    print(f"Collected {len(df)} metric rows across {n_seeds} seed(s) -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect per-run/per-model metrics JSON files into one Parquet table."
    )
    parser.add_argument("--task", type=str, required=True, choices=["top", "quark-gluon"])
    main(parser.parse_args())
