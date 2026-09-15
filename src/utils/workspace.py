# workspace.py
import os
import json
import datetime
import numpy as np
import torch
import random
from pathlib import Path

from src.utils.hyperparams import get_hyperparams


def get_project_root():
    """
    Returns the absolute path to the project root directory.
    This is useful for constructing paths to data, models, and reports in a way that is independent of the current working directory.
    """
    return Path(__file__).parent.parent.parent.resolve()

# Set random seeds for reproducibility
def set_seed(seed_value=42, purpose=None):
    suffix = f" ({purpose})" if purpose else ""
    print(f"Setting global random seed {seed_value} for reproducibility{suffix}.")
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

def write_hyperparams_snapshot(config, extra=None):
    """
    Serializes this run's resolved hyperparameters to config["hyperparams_report_path"]
    as JSON: task, seed, a timestamp, the subset of `config` matching get_hyperparams()'s
    keys (so it reflects what this config object actually resolved to), and an optional
    `extra` dict of script-identifying info (e.g. parsed CLI args) under 'run_args'.
    """
    hp_keys = get_hyperparams().keys()
    snapshot = {
        "task": config.get("task"),
        "seed": config.get("seed"),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "hyperparams": {k: config[k] for k in hp_keys if k in config},
        "run_args": extra or {},
    }
    path = Path(config["hyperparams_report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)

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

    # Seed-INDEPENDENT directories: the canonical, build-once 15-way disjoint
    # partition (shared across every seed/subset run) and the cross-run metrics
    # collection table both live at the task level, not nested under seed_<seed>/.
    canonical_dir = os.path.join(root, "data", "processed", task, "canonical")
    aggregate_dir = os.path.join(root, "outputs", task, "aggregate")

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

        # Canonical (seed-independent) 15-way disjoint subset partition -- built
        # once by scripts/run_preprocessing.py / run_preprocessing_qg.py, only ever
        # read (never rebuilt) by the training scripts.
        "canonical_data_dir": canonical_dir,
        "canonical_cache_file": os.path.join(canonical_dir, "preprocessed_subsets.pt"),
        "canonical_scaler_path": os.path.join(canonical_dir, "global_scaler.pkl"),

        # Cross-run metrics collection (Parquet table, task-level)
        "aggregate_dir": aggregate_dir,
        "metrics_table_path": os.path.join(aggregate_dir, "metrics_table.parquet"),

        # Output targets for models and reports
        "models_dir": os.path.join(outputs_dir, "models", "01_base"),
        "plots_dir": os.path.join(outputs_dir, "plots"),
        "results_dir": os.path.join(outputs_dir, "results"),
        "logs_dir": os.path.join(outputs_dir, "logs"),
        "hyperparams_report_path": os.path.join(outputs_dir, "hyperparameters.json"),

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
        "base_eval_cm_normalized": os.path.join(outputs_dir, "plots", "01_plot_base", "base_eval_cm_normalized.png"),
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
        "retrain_eval_cm_normalized": os.path.join(outputs_dir, "plots", "03_retrained", "retrain_eval_cm_normalized.png"),
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
        "symbolic_eval_cm_normalized": os.path.join(outputs_dir, "plots", "04_symbolic", "symbolic_eval_cm_normalized.png"),
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
        "final_eval_cm_normalized": os.path.join(outputs_dir, "plots", "05_final", "final_eval_cm_normalized.png"),
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

        # ----------------------------
        # --- Report quantum Paths ---
        # ----------------------------
        # Chebyshev coefficients txt paths
        "Chebyshev_coefficients_path": os.path.join(outputs_dir, "results", "chebyshev_coefficients.txt"),
        "circuit_plot": os.path.join(outputs_dir, "plots", "quantum-circuit.png"),
        "qkan_metadata_path": os.path.join(outputs_dir, "results", "qkan_metadata.json"),

        # Reports - QKAN evaluation - Noisy
        "roc_qkan_noisy": os.path.join(outputs_dir, "plots", "qkan","noisy", "roc_qkan_noisy.png"),
        "pr_qkan_noisy": os.path.join(outputs_dir, "plots", "qkan","noisy", "pr_qkan_noisy.png"),
        "cm_qkan_noisy": os.path.join(outputs_dir, "plots", "qkan","noisy", "cm_qkan_noisy.png"),
        "cm_qkan_noisy_normalized": os.path.join(outputs_dir, "plots", "qkan","noisy", "cm_qkan_noisy_normalized.png"),
        "metrics_qkan_noisy": os.path.join(outputs_dir, "results", "qkan","noisy", "metrics_qkan_noisy.json"),
        "history_noisy_loss": os.path.join(outputs_dir, "results", "qkan","noisy", "history_loss.json"),
        "history_noisy_loss_plot": os.path.join(outputs_dir, "plots", "qkan","noisy", "history_loss.png"),
        "history_noisy_auc_plot": os.path.join(outputs_dir, "plots", "qkan","noisy", "history_auc.png"),

        # Reports - QKAN evaluation - Ideal
        "roc_qkan_ideal": os.path.join(outputs_dir, "plots", "qkan","ideal", "roc_qkan_ideal.png"),
        "pr_qkan_ideal": os.path.join(outputs_dir, "plots", "qkan","ideal", "pr_qkan_ideal.png"),
        "cm_qkan_ideal": os.path.join(outputs_dir, "plots", "qkan","ideal", "cm_qkan_ideal.png"),
        "cm_qkan_ideal_normalized": os.path.join(outputs_dir, "plots", "qkan","ideal", "cm_qkan_ideal_normalized.png"),
        "metrics_qkan_ideal": os.path.join(outputs_dir, "results", "qkan","ideal", "metrics_qkan_ideal.json"),
        "history_ideal_loss": os.path.join(outputs_dir, "results", "qkan","ideal", "history_loss.json"),
        "history_ideal_loss_plot": os.path.join(outputs_dir, "plots", "qkan","ideal", "history_loss.png"),
        "history_ideal_auc_plot": os.path.join(outputs_dir, "plots", "qkan","ideal", "history_auc.png"),

        # Reports - QKAN evaluation - Shots
        "roc_qkan_shots": os.path.join(outputs_dir, "plots", "qkan","shots", "roc_qkan_shots.png"),
        "pr_qkan_shots": os.path.join(outputs_dir, "plots", "qkan","shots", "pr_qkan_shots.png"),
        "cm_qkan_shots": os.path.join(outputs_dir, "plots", "qkan","shots", "cm_qkan_shots.png"),
        "cm_qkan_shots_normalized": os.path.join(outputs_dir, "plots", "qkan","shots", "cm_qkan_shots_normalized.png"),
        "metrics_qkan_shots": os.path.join(outputs_dir, "results", "qkan","shots", "metrics_qkan_shots.json"),
        "history_shots_loss": os.path.join(outputs_dir, "results", "qkan","shots", "history_loss.json"),
        "history_shots_loss_plot": os.path.join(outputs_dir, "plots", "qkan","shots", "history_loss.png"),
        "history_shots_auc_plot": os.path.join(outputs_dir, "plots", "qkan","shots", "history_auc.png"),

        # Reports - QKAN evaluation - Baseline (untrained, warm-start only, pre-training)
        "roc_qkan_baseline_noisy": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "roc_qkan_baseline_noisy.png"),
        "pr_qkan_baseline_noisy": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "pr_qkan_baseline_noisy.png"),
        "cm_qkan_baseline_noisy": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "cm_qkan_baseline_noisy.png"),
        "cm_qkan_baseline_noisy_normalized": os.path.join(outputs_dir, "plots", "qkan", "noisy", "baseline", "cm_qkan_baseline_noisy_normalized.png"),
        "metrics_qkan_baseline_noisy": os.path.join(outputs_dir, "results", "qkan", "noisy", "baseline", "metrics_qkan_baseline_noisy.json"),

        "roc_qkan_baseline_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "roc_qkan_baseline_ideal.png"),
        "pr_qkan_baseline_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "pr_qkan_baseline_ideal.png"),
        "cm_qkan_baseline_ideal": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "cm_qkan_baseline_ideal.png"),
        "cm_qkan_baseline_ideal_normalized": os.path.join(outputs_dir, "plots", "qkan", "ideal", "baseline", "cm_qkan_baseline_ideal_normalized.png"),
        "metrics_qkan_baseline_ideal": os.path.join(outputs_dir, "results", "qkan", "ideal", "baseline", "metrics_qkan_baseline_ideal.json"),

        "roc_qkan_baseline_shots": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "roc_qkan_baseline_shots.png"),
        "pr_qkan_baseline_shots": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "pr_qkan_baseline_shots.png"),
        "cm_qkan_baseline_shots": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "cm_qkan_baseline_shots.png"),
        "cm_qkan_baseline_shots_normalized": os.path.join(outputs_dir, "plots", "qkan", "shots", "baseline", "cm_qkan_baseline_shots_normalized.png"),
        "metrics_qkan_baseline_shots": os.path.join(outputs_dir, "results", "qkan", "shots", "baseline", "metrics_qkan_baseline_shots.json"),

        "init_weights": os.path.join(data_out_dir, "quantum_weights"),

        # Quantum model and weights
        "polynomial_weights_dir": os.path.join(data_out_dir, "quantum_weights"),
        "coef_n_path": os.path.join(data_out_dir, "quantum_weights", "w_n.npy"),
        "coef_q_path": os.path.join(data_out_dir, "quantum_weights", "w_q.npy"),
        "coef_z_path": os.path.join(data_out_dir, "quantum_weights", "w_z.npy"),
        "coef_dr_path": os.path.join(data_out_dir, "quantum_weights", "w_dr.npy"),
        "coef_out_path": os.path.join(data_out_dir, "quantum_weights", "w_out.npy"),

        "qkan_noisy_path": os.path.join(data_out_dir, "quantum_weights", "qkan_noisy.pth"),
        "qkan_ideal_path": os.path.join(data_out_dir, "quantum_weights", "qkan_ideal.pth"),
        "qkan_shots_path": os.path.join(data_out_dir, "quantum_weights", "qkan_shots.pth"),

        # ----------------------------------
        # --- Random Forest baseline paths ---
        # ----------------------------------
        "rf_model_path": os.path.join(outputs_dir, "models", "rf", "rf_model.joblib"),
        "rf_eval_data_true": os.path.join(outputs_dir, "results", "rf", "rf_eval_true.npy"),
        "rf_eval_data_probs": os.path.join(outputs_dir, "results", "rf", "rf_eval_probs.npy"),
        "rf_eval_data_binary": os.path.join(outputs_dir, "results", "rf", "rf_eval_binary.npy"),
        "rf_eval_metrics": os.path.join(outputs_dir, "results", "rf", "rf_eval_metrics.json"),
        "rf_feature_importance_data": os.path.join(outputs_dir, "results", "rf", "rf_feature_importance.json"),
        "rf_eval_cm": os.path.join(outputs_dir, "plots", "rf", "rf_eval_cm.png"),
        "rf_eval_cm_normalized": os.path.join(outputs_dir, "plots", "rf", "rf_eval_cm_normalized.png"),
        "rf_eval_roc": os.path.join(outputs_dir, "plots", "rf", "rf_eval_roc.png"),
        "rf_eval_pr": os.path.join(outputs_dir, "plots", "rf", "rf_eval_pr.png"),
        "rf_feature_importance_plot": os.path.join(outputs_dir, "plots", "rf", "rf_feature_importance.png"),
    }

    CONFIG.update(get_hyperparams())
    return CONFIG
