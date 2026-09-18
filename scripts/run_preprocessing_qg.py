import sys
from pathlib import Path
import os
import argparse

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.workspace import get_config
from src.preprocessing.processor_qg import load_and_preprocess_data

def main(full_dataset=False):
    # Load global configuration (task and seed); full_dataset forces n_subsets=1
    config = get_config(task="quark-gluon", seed=42, full_dataset=full_dataset)
    qg_path = os.path.join(config["raw_data_dir"], "quark-gluon")  # Path to the raw Quark-Gluon data
    print(f"Preprocessing regime: {config['variant']} (n_subsets={config['n_subsets']}) "
          f"-> {config['canonical_data_dir']}")
    # Run the pipeline
    # This automatically checks the cache, if it doesn't exist, processes and saves
    X_train, y_train, X_val, y_val, X_test, y_test, X_sample, scaler = load_and_preprocess_data(
        data_dir=qg_path,
        task="quark-gluon",
        force_process=True,
        full_dataset=full_dataset,
    )
    
    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quark-gluon raw-to-canonical preprocessing pipeline.")
    parser.add_argument(
        "--full-dataset",
        dest="full_dataset",
        action="store_true",
        help="Use the entire dataset (n_subsets=1; train/val/test stay separate). Default: off.",
    )
    args = parser.parse_args()
    main(full_dataset=args.full_dataset)
