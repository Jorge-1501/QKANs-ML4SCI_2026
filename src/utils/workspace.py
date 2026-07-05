# workspace.py
import os
import numpy as np
import torch
import random
from pathlib import Path

features_globales = ['m', 'n']
features_locales = []
for i in range(1, 11):
    features_locales.append(rf'DR_{i}')
    features_locales.append(rf'pT_{i}')

TOTAL_FEATURES = features_globales + features_locales

def get_project_root():
    """
    Returns the absolute path to the project root directory.
    This is useful for constructing paths to data, models, and reports in a way that is independent of the current working directory.
    """
    return Path(__file__).parent.parent.parent.resolve()

# Set random seeds for reproducibility
def set_seed(seed_value=42):
    print(f"Setting global random seed {seed_value} for reproducibility.")
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def make_dirs(config):
    """
    Creates all necessary directories based on the provided configuration dictionary.
    """
    print("Ensuring directory structure exists...")
    for key, value in config.items():
        if isinstance(value, (str, Path)) and str(config['root']) in str(value):
            path=Path(value)
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# STEP 1: CONFIGURATION
# ============================================================================
# Define all hyperparameters and paths in one place.
# This makes it easy to modify and experiment.
def get_config(task, seed):
    """
    Returns a tight configuration dictionary isolating the raw data path, 
    the processed multiscale tensors, and specific KAN 2.0 / VQC output targets.
    """
    root = get_project_root()
    seed_dir = f"seed_{seed}"
    
    # Core directories
    data_out_dir = os.path.join(root, "data", "processed", task, seed_dir)
    outputs_dir = os.path.join(root, "outputs", task, seed_dir)
    CONFIG = {
        # Base Engine Paths
        "root": root,
        "task": task,
        "seed": seed,

        # Origin and Destination of Data
        "raw_data_dir": os.path.join(root, "data", "raw"),
        "processed_data_dir": data_out_dir,
        "scaler_path": os.path.join(data_out_dir, "global_scaler.pkl"),
        "cache_file": os.path.join(data_out_dir, "preprocessed_data.pt"),
        
        # Output targets for models and reports
        "models_dir": os.path.join(outputs_dir, "models", "01_base"),
        "plots_dir": os.path.join(outputs_dir, "plots"),
        "results_dir": os.path.join(outputs_dir, "results"),

        # reports
        "base_train_history_data": os.path.join(outputs_dir, "results", "01_base", "base_train_history.json"),
        "base_eval_data_true": os.path.join(outputs_dir, "results", "01_base", "base_eval_true.npy"),
        "base_eval_data_probs": os.path.join(outputs_dir, "results", "01_base", "base_eval_probs.npy"),
        "base_eval_data_binary": os.path.join(outputs_dir, "results", "01_base", "base_eval_binary.npy"),
        "base_eval_metrics": os.path.join(outputs_dir, "results", "01_base", "base_eval_metrics.json"),

        # plots
        "base_train_loss_plot": os.path.join(outputs_dir, "plots", "01_plot_base", "base_train_loss.png"),
        "base_train_auc_plot": os.path.join(outputs_dir, "plots", "01_plot_base", "base_train_auc.png"),
        "base_eval_cm": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_cm.png"),
        "base_eval_roc": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_roc.png"),
        "base_eval_pr": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_pr.png"),

        "base_model_plot_folder": os.path.join(outputs_dir, "plots", "01_plot_base", "splines"),
        "base_model_plot_save_path": os.path.join(outputs_dir, "plots", "01_plot_base", "base_model.png"),
        
        # Pruned
        "pruned_model_path": os.path.join(outputs_dir, "models", "02_pruned"),

        # Re-trained
        "retrained_model_path": os.path.join(outputs_dir, "models", "03_retrained"),
        "retrain_history_data": os.path.join(outputs_dir, "results", "03_retrained", "retrain_history.json"),
        "retrain_loss_plot": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_loss.png"),
        "retrain_auc_plot": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_auc.png"),
        "retrain_eval_cm": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_cm.png"),
        "retrain_eval_roc": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_roc.png"),
        "retrain_eval_pr": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_pr.png"),
        "retrain_eval_data_true": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_true.npy"),
        "retrain_eval_data_probs": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_probs.npy"),
        "retrain_eval_data_binary": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_binary.npy"),
        "retrain_eval_metrics": os.path.join(outputs_dir, "results", "03_retrained", "retrain_eval_metrics.json"),
        "retrained_model_plot_folder": os.path.join(outputs_dir, "plots", "03_retrained", "splines"),
        "retrained_model_plot_save_path": os.path.join(outputs_dir, "plots", "03_retrained", "retrained_model.png"),

        # symbolic simplification
        "symbolic_model_path": os.path.join(outputs_dir, "models", "04_symbolic"),
        "symbolic_model_plot_folder": os.path.join(outputs_dir, "plots", "04_symbolic", "splines"),
        "symbolic_model_plot_save_path": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_model.png"),
        "symbolic_model_eval_data": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_model_eval_data.json"),
        "symbolic_eval_cm": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_cm.png"),
        "symbolic_eval_roc": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_roc.png"),
        "symbolic_eval_pr": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_pr.png"),
        "symbolic_eval_data_true": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_true.npy"),
        "symbolic_model_eval_probs": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_probs.npy"),
        "symbolic_model_eval_binary": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_binary.npy"),
        "symbolic_eval_metrics": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_eval_metrics.json"),
        "symbolic_history_data": os.path.join(outputs_dir, "results", "04_symbolic", "symbolic_history.json"),
        "symbolic_loss_plot": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_loss.png"),
        "symbolic_auc_plot": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_auc.png"),

        # final fine-tuning
        "final_model_path": os.path.join(outputs_dir, "models", "05_final"),
        "final_eval_cm": os.path.join(outputs_dir, "plots", "05_final", "final_eval_cm.png"),
        "final_eval_roc": os.path.join(outputs_dir, "plots", "05_final", "final_eval_roc.png"),
        "final_eval_pr": os.path.join(outputs_dir, "plots", "05_final", "final_eval_pr.png"),
        "final_eval_data_true": os.path.join(outputs_dir, "results", "05_final", "final_eval_true.npy"),
        "final_eval_data_probs": os.path.join(outputs_dir, "results", "05_final", "final_eval_probs.npy"),
        "final_eval_data_binary": os.path.join(outputs_dir, "results", "05_final", "final_eval_binary.npy"),
        "final_eval_metrics": os.path.join(outputs_dir, "results", "05_final", "final_eval_metrics.json"),
        "final_formula_path": os.path.join(outputs_dir, "results", "05_final", "final_formula.txt"),
        "final_model_plot_folder": os.path.join(outputs_dir, "plots", "05_final", "splines"),
        "final_model_plot_save_path": os.path.join(outputs_dir, "plots", "05_final", "final_model.png"),
        "final_history_data": os.path.join(outputs_dir, "results", "05_final", "final_history.json"),
        "final_loss_plot": os.path.join(outputs_dir, "plots", "05_final", "final_loss.png"),
        "final_auc_plot": os.path.join(outputs_dir, "plots", "05_final", "final_auc.png"),

        # -----------------------------
        # --- Classic KAN ----
        # -----------------------------
        "features": TOTAL_FEATURES,
        "width": [22, [9,9], 1], # Architecture [input, hidden, output]
        "grid": 3,
        "k": 3,
        "num_workers": 0,

        # --- Base Training Hyperparameters ---
        "base_lr": 1e-3,
        "base_epochs": 50,
        "base_batch_size": 4096,
        "base_patience": 7,
        "base_early_stop_delta": 5e-3,
        "base_lamb": 0.01, # Regularization weight
        "base_lamb_l1": 0.1,
        "base_lamb_entropy": 0.2,
        "base_lamb_coef": 0.005,     
        "base_lamb_coefdiff": 0.01,
        "base_update_grid_freq": 50,

        # --- Pruning Hyperparameters ---
        "prune_node_th": 1e-2,
        "prune_edge_th": 3e-2,

        # --- Re-training Hyperparameters ---
        "retrain_lr": 1e-3,
        "retrain_epochs": 20,
        "retrain_batch_size": 8192,
        "retrain_patience": 6,
        "retrain_early_stop_delta": 5e-5,
        "retrain_lamb_l1": 0.1,
        "retrain_lamb_entropy": 0.2,
        "retrain_lamb_coef": 0.005,
        "retrain_lamb_coefdiff": 0.01,

        # --- Simplification (Fitting) Hyperparameters ---
        "symbolic_r2_threshold": 0.8,
        "symbolic_weight_simple": 0.5,

        # --- Fine-Tuning Hyperparameters ---
        "finetune_lr": 5e-5, # Lower learning rate
        "finetune_epochs": 15,
        "finetune_batch_size": 512,
        "finetune_patience": 5,
        "finetune_early_stop_delta": 1e-6
    }
    return CONFIG
