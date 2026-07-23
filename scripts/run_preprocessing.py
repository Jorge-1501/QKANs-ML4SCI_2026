import sys
from pathlib import Path
import os
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils.workspace import get_config
from src.preprocessing.processor_top import load_and_preprocess_data

def main():
    # Load global configuration (task and seed)
    config = get_config(task="top", seed=42)
    top_path = os.path.join(config["raw_data_dir"], "top")
    # Run the pipeline
    # This automatically checks the cache, if it doesn't exist, processes and saves
    X_train, y_train, X_val, y_val, X_test, y_test, X_sample, scaler = load_and_preprocess_data(
        data_dir=top_path,
        processed_dir=config["processed_data_dir"],
        task="top",
        force_process=True
    )
    
    print("Preprocessing pipeline completed successfully.")

if __name__ == "__main__":
    main()
