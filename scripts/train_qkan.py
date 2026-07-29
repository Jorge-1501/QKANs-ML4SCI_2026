# train_qkan.py
import argparse
import traceback
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
import src.preprocessing.processor_top as processor
from src.architectures.extractor import SymbolicWarmStartExtractor
from src.architectures.quantum_kan import QuantumKANTrainer

class TeedLog:
    """Clone sys.stdout to log messages to both the console and a file."""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message) 
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def main(args):
    # Configuration and workspace setup
    CONFIG = workspace.get_config(task=args.task, seed=args.seed)
    workspace.make_dirs(CONFIG)
    print(f"Selected backend mode (Training): {args.train_backend} \n")

    log_file_path = os.path.join(CONFIG["processed_data_dir"], f"train_qkan_{args.task}.log")
    sys.stdout = TeedLog(log_file_path)

    # Load classical data
    X_train, y_train, X_val, y_val, X_test, y_test, X_sample, scaler = processor.load_and_preprocess_data(
        data_dir=os.path.join("data", "raw", args.task),
        processed_dir=CONFIG["processed_data_dir"],
        task=args.task,
        force_process=False,
        seed=args.seed
    )

    # Automatic Extraction (Agnostic Warm-Start)
    extractor = SymbolicWarmStartExtractor(CONFIG)
    
    classic_model_path = os.path.join(CONFIG['final_model_path'], "05_final")
    output_weights_path = os.path.join(CONFIG["polynomial_weights_dir"], "quantum_weights.pt")
    report_path = CONFIG.get("Chebyshev_coefficients_path", os.path.join(CONFIG["results_dir"], "chebyshev_report.txt"))
    
    # If forced or not existing, extract classical model weights
    if args.force or not os.path.exists(output_weights_path):
        extractor.extract_and_save(classic_model_path, output_weights_path, report_path)
    else:
        print(f"Initial Weights exist in {output_weights_path}. Skipping extraction.")

    # Quantum Initialization and Training
    q_trainer = QuantumKANTrainer(CONFIG, train_backend=args.train_backend)
    
    # Plot the dynamically generated circuit before training
    q_trainer.model.plot_circuit(CONFIG.get("circuit_plot", os.path.join(CONFIG["plots_dir"], "quantum-circuit.png")))
    
    # Quantum optimization loop
    history = q_trainer.fit(X_train, y_train, X_val, y_val, resume=True, force=args.force)

    # Final Evaluation
    # Now calling the internal method of the class
    q_trainer.evaluate(X_test, y_test, eval_backend=args.eval_backend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train QKAN with architecture inferred from the classical model.")
    parser.add_argument('--seed', type=int, default=42, help='Global seed')
    parser.add_argument('--train_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='ideal')
    parser.add_argument('--eval_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='noisy')
    parser.add_argument('--force', action='store_true', help='Force extraction and retraining')
    parser.add_argument('--task', type=str, choices=['top', 'quark-gluon'], default='top')
    args = parser.parse_args()
    
    try:
        main(args)
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        traceback.print_exc()