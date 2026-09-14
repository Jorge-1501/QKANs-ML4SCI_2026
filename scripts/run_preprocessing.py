import sys
from pathlib import Path
import os
import argparse
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.workspace import get_config
from src.preprocessing.processor_top import load_and_preprocess_data

def main():
    parser = argparse.ArgumentParser(description="Top-tagging raw-to-canonical preprocessing pipeline.")
    parser.add_argument(
        "--apply-mass-cut",
        dest="apply_mass_cut",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply the invariant-mass cut (bounds from hyperparams.py's "
            "mass_cut_lo/mass_cut_hi). Default: hyperparams.py's apply_mass_cut "
            "value. Pass --no-apply-mass-cut to use all data, no mass cut."
        ),
    )
    args = parser.parse_args()

    # Load global configuration (task and seed)
    config = get_config(task="top", seed=42)
    top_path = os.path.join(config["raw_data_dir"], "top")
    # Run the pipeline
    # This automatically checks the cache, if it doesn't exist, processes and saves
    X_train, y_train, X_val, y_val, X_test, y_test, X_sample, scaler = load_and_preprocess_data(
        data_dir=top_path,
        task="top",
        force_process=True,
        apply_mass_cut=args.apply_mass_cut
    )

    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    main()
