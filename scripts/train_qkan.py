# train_qkan.py
import argparse
import traceback
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
import src.preprocessing.processor_qg as processor
from src.architectures.quantum_kan import QuantumKANTrainer
from src.utils.evaluate_qkan import evaluate_qkan

def main(args):
    """
    """
    # Load configuration and create necessary directories
    CONFIG = workspace.get_config(task=args.task, seed=args.seed)
    workspace.make_dirs(CONFIG)

    print(f"Selected backend mode: {args.train_backend} \n")

    # Load and preprocess the data for the specified task
    X_train, y_train, X_val, y_val, _, _, _, _ = processor.load_and_preprocess_data(
        data_dir=f"data/raw/{args.task}",
        processed_dir=CONFIG["processed_data_dir"],
        task=args.task,
        force_process=False
    )

    # Initialize the Quantum KAN Trainer with the specified backend
    q_trainer = QuantumKANTrainer(CONFIG, train_backend=args.train_backend)
    
    # Train the model (the method encapsulates fault tolerance/checkpoints per file)
    history = q_trainer.fit(X_train, y_train, X_val, y_val, force=args.force)

    # Final evaluation delegated to a separate function for clarity and modularity
    evaluate_qkan(CONFIG, train_backend=args.train_backend, eval_backend=args.eval_backend)

# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the QKAN quantum model with warm-started weights.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='ideal', help='Quantum backend mode to use for training')
    parser.add_argument('--eval_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='noisy', help='Quantum backend mode to use for evaluation')
    parser.add_argument('--force', action='store_true', help='Force retraining even if model already exists')
    parser.add_argument('--task', type=str, choices=['top', 'quark-gluon'], default='quark-gluon', help='Task to perform')
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()