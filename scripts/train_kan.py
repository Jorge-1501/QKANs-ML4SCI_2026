import sys
from pathlib import Path
# Añadir la raíz al path para importar src y workspace
sys.path.append(str(Path(__file__).parent.parent.resolve()))
import src.utils.workspace as workspace
import src.preprocessing.processor_top as processor
import src.architectures.classic_kan as classic
import src.utils.metrics as viz
from src.architectures.classic_kan import clean_memory

import torch
import time
import os
import json
import numpy as np
import argparse
import gc
from pathlib import Path

def main(args):
    """Main fuction to run the full training pipeline."""
    torch.set_num_threads(4)  # Limit PyTorch to use 4 CPU threads for data loading and processing
    workspace.set_seed(args.seed)
    
    CONFIG = workspace.get_config(task="top", seed=args.seed)
    workspace.make_dirs(CONFIG)
    
    start_time = time.time()
    print(f"Starting the automated training pipeline. Seed: {args.seed}.")


    # ============================================================================
    # STEP 1: DATA LOADING AND PREPROCESSING
    # ============================================================================
    print("\n--- Step 1: Loading and Preprocessing Data ---")
    top_path = os.path.join(CONFIG["raw_data_dir"], "top")
    X_train, y_train, \
    X_val, y_val, \
    X_test, y_test, \
    X_sample, scaler = processor.load_and_preprocess_data(
                                        data_dir=top_path,
                                        processed_dir=CONFIG["processed_data_dir"],
                                        task=CONFIG["task"],
                                        force_process=False
                                    )
    del top_path
    gc.collect()
    
    # Saving scaler to inference

    # ============================================================================
    # STEP 2: BASE TRAINING
    # ============================================================================
    print("\n--- Step 2: Starting Base Training ---")
    base_model_prefix = os.path.join(CONFIG["models_dir"], "01_base")
    base_model_state_path = f"{base_model_prefix}_state"

    if os.path.exists(base_model_state_path) and not args.force:
        print(f"Base model found at {base_model_prefix}")
    else:
        history_base = classic.train_kan_model(
            width=CONFIG["width"],
            grid=CONFIG["grid"],
            k=CONFIG["k"],
            learning_rate=CONFIG["base_lr"],
            num_epochs=CONFIG["base_epochs"],
            batch_size=CONFIG["base_batch_size"], 
            early_stop_patience=CONFIG["base_patience"],
            early_stop_min_delta=CONFIG["base_early_stop_delta"],
            lamb=CONFIG["base_lamb"],
            lamb_l1=CONFIG["base_lamb_l1"],
            lamb_entropy=CONFIG["base_lamb_entropy"],
            lamb_coef=CONFIG["base_lamb_coef"],
            lamb_coefdiff=CONFIG["base_lamb_coefdiff"],
            update_grid_freq=CONFIG["base_update_grid_freq"],
            model_save_path=base_model_prefix,
            X_train_tensor=X_train[:],
            y_train_tensor=y_train[:],
            X_val_tensor=X_val[:10000], 
            y_val_tensor=y_val[:10000],
            num_workers=CONFIG["num_workers"]
        )
        
        with open(CONFIG["base_train_history_data"], 'w') as f:
            json.dump(history_base, f, indent=4)

        viz.plot_loss_history(history_base, save_path=CONFIG["base_train_loss_plot"])
        viz.plot_auc_history(history_base, save_path=CONFIG["base_train_auc_plot"])

        print("\n--- Evaluation of Base Model ---")
        model_base, eval_data_base, metrics_base = classic.evaluate_kan_model(
            model_save_path=base_model_prefix,
            X_test_tensor=X_test[:10000],
            y_test_tensor=y_test[:10000],
            conf_matrix_save_path=CONFIG["base_eval_cm"],
            save_path_roc_curve=CONFIG["base_eval_roc"],
            save_path_pr_curve=CONFIG["base_eval_pr"]
        )

        np.save(CONFIG["base_eval_data_true"], eval_data_base[0])
        np.save(CONFIG["base_eval_data_probs"], eval_data_base[1])
        np.save(CONFIG["base_eval_data_binary"], eval_data_base[2])

        with open(CONFIG["base_eval_metrics"], 'w') as f:
            json.dump(metrics_base, f, indent=4)

        print("\n--- Plotting Base Model Splines ---")
        model_base.plot(
            folder=CONFIG["base_model_plot_folder"],
            save_path=CONFIG["base_model_plot_save_path"],
            beta=12.0,
            metric="backward",
            in_vars=CONFIG["features"],
            scale=1.0,
            varscale=0.5
        )

        print("Cleaning up base model from memory...")
        clean_memory(model_base, eval_data_base, metrics_base, history_base)

# ============================================================================
# STEP 3: PRUNING
# ============================================================================
    print("\n--- Step 3: Starting Pruning ---")
    pruned_model_prefix = os.path.join(CONFIG["pruned_model_path"], "02_pruned")
    pruned_model_state_path = f"{pruned_model_prefix}_state"

    if os.path.exists(pruned_model_state_path) and not args.force:
        print(f"Pruned model found. Skiping pruning.")
    else:
        pruned_model = classic.prune_and_save_kan(
            original_model_path=base_model_prefix,
            pruned_model_path=pruned_model_prefix,
            activation_data=X_sample, # Use sample for fast pruning
            node_th=CONFIG["prune_node_th"],
            edge_th=CONFIG["prune_edge_th"]
        )
        # del pruned_model
        gc.collect()

# ============================================================================
# STEP 4: POST-PRUNING RETRAINING
# ============================================================================
    print("\n--- Step 4: Starting Post-Pruning Retraining ---")

    retrained_model_prefix = os.path.join(CONFIG["retrained_model_path"], "03_retrained")
    retrained_model_state_path = f"{retrained_model_prefix}_state"
    
    if os.path.exists(retrained_model_state_path) and not args.force:
        print(f"Skipping retraining. Model found")
    else:
        # Passing the 'pruned_model' object returned from the previous function
        final_retrained_model, history_retrain = classic.retrain_pruned_kan(
            pruned_model=pruned_model,
            learning_rate=CONFIG["retrain_lr"], num_epochs=CONFIG["retrain_epochs"],
            batch_size=CONFIG["retrain_batch_size"],
            early_stop_patience=CONFIG["retrain_patience"],
            early_stop_min_delta=CONFIG["retrain_early_stop_delta"],
            model_save_path=retrained_model_prefix,
            X_train_tensor=X_train, y_train_tensor=y_train,
            X_val_tensor=X_val, y_val_tensor=y_val,
            lamb=0.001,
            lamb_l1=CONFIG["retrain_lamb_l1"],
            lamb_entropy=CONFIG["retrain_lamb_entropy"],
            lamb_coef=CONFIG["retrain_lamb_coef"],
            lamb_coefdiff=CONFIG["retrain_lamb_coefdiff"],
            num_workers=CONFIG["num_workers"]
        )

        with open(CONFIG["retrain_history_data"], 'w') as f:
            json.dump(history_retrain, f, indent=4)

        viz.plot_loss_history(history_retrain, save_path=CONFIG["retrain_loss_plot"])
        viz.plot_auc_history(history_retrain, save_path=CONFIG["retrain_auc_plot"])
        
        print("\n--- Evaluation of Retrained Model ---")
        model_retrained, eval_data_retrained, metrics_retrained = classic.evaluate_kan_model(
            model_save_path=retrained_model_prefix,
            X_test_tensor=X_test, y_test_tensor=y_test,
            conf_matrix_save_path=CONFIG["retrain_eval_cm"],
            save_path_roc_curve=CONFIG["retrain_eval_roc"],
            save_path_pr_curve=CONFIG["retrain_eval_pr"]
        )

        np.save(CONFIG["retrain_eval_data_true"], eval_data_retrained[0])
        np.save(CONFIG["retrain_eval_data_probs"], eval_data_retrained[1])
        np.save(CONFIG["retrain_eval_data_binary"], eval_data_retrained[2])

        with open(CONFIG["retrain_eval_metrics"], 'w') as f:
            json.dump(metrics_retrained, f, indent=4)

        print("\n--- Plotting Retrained Model Splines ---")
        model_retrained.plot(
            folder=CONFIG["retrained_model_plot_folder"],
            save_path=CONFIG["retrained_model_plot_save_path"],
            beta=12.0,
            metric="backward",
            in_vars=CONFIG["features"],
            scale=1.0,
            varscale=0.5
        )

        clean_memory(model_retrained, eval_data_retrained, history_retrain, metrics_retrained)

# ============================================================================
# STEP 5: SYMBOLIC SIMPLIFICATION (FITTING)
# ============================================================================
    print("\n--- Step 5: Starting Symbolic Simplification ---")
    symbolic_model_prefix = os.path.join(CONFIG["symbolic_model_path"], "04_symbolic")
    symbolic_model_state_path = f"{symbolic_model_prefix}_state"
    if os.path.exists(symbolic_model_state_path) and not args.force:
        print(f"Symbolic model found, skiping simplification.")

    else:
        classic.simplify_and_save(
            source_checkpoint_path=retrained_model_prefix,
            symbolic_model_path=symbolic_model_prefix,
            x_train_sample=X_sample, # Use sample for fitting
            r2_threshold=CONFIG["symbolic_r2_threshold"],
            weight_simple=CONFIG["symbolic_weight_simple"]
        )

        model_symbolic, eval_data_symbolic, metrics_symbolic = classic.evaluate_kan_model(
            model_save_path=symbolic_model_prefix,
            X_test_tensor=X_test, y_test_tensor=y_test,
            conf_matrix_save_path=CONFIG["symbolic_eval_cm"],
            save_path_roc_curve=CONFIG["symbolic_eval_roc"],
            save_path_pr_curve=CONFIG["symbolic_eval_pr"]
        )

        np.save(CONFIG["symbolic_model_eval_data"], eval_data_symbolic[0])
        np.save(CONFIG["symbolic_model_eval_probs"], eval_data_symbolic[1])
        np.save(CONFIG["symbolic_model_eval_binary"], eval_data_symbolic[2])

        with open(CONFIG["symbolic_eval_metrics"], 'w') as f:
            json.dump(metrics_symbolic, f, indent=4)

        print("\n--- Plotting Symbolic Model Splines ---")
        model_symbolic.plot(
            folder=CONFIG["symbolic_model_plot_folder"],
            save_path=CONFIG["symbolic_model_plot_save_path"],
            beta=12.0,
            metric="backward",
            in_vars=CONFIG["features"],
            scale=1.0,
            varscale=0.5
        )

        clean_memory(model_symbolic, eval_data_symbolic, metrics_symbolic)

# ============================================================================
# STEP 6: SYMBOLIC FINE-TUNING
# ============================================================================
    print("\n--- Step 6: Starting Symbolic Fine-Tuning ---")

    final_model_prefix = os.path.join(CONFIG["final_model_path"], "05_final")
    final_model_state_path = f"{final_model_prefix}_state"
    
    if os.path.exists(final_model_state_path) and not args.force:
        print("Symbolic model with finetune found. Skipping step.")
    else:
        _, history_finetune = classic.finetune_symbolic_model(
            source_symbolic_path=symbolic_model_prefix,
            final_model_path=final_model_prefix,
            X_train_tensor=X_train,
            y_train_tensor=y_train,
            X_val_tensor=X_val, y_val_tensor=y_val,
            learning_rate=CONFIG["finetune_lr"],
            num_epochs=CONFIG["finetune_epochs"],
            batch_size=CONFIG["finetune_batch_size"],
            early_stop_patience=CONFIG["finetune_patience"],
            early_stop_min_delta=CONFIG["finetune_early_stop_delta"],
            num_workers=CONFIG["num_workers"]
        )

        with open(CONFIG["final_history_data"], 'w') as f:
            json.dump(history_finetune, f, indent=4)

        viz.plot_loss_history(history_finetune, save_path=CONFIG["final_loss_plot"])
        viz.plot_auc_history(history_finetune, save_path=CONFIG["final_auc_plot"])

        # Cleaning model and data from memory before final evaluation
        clean_memory(history_finetune, 
                 X_train, y_train, 
                 X_val, y_val,
                 eval_data_symbolic, metrics_symbolic)

# ============================================================================
# STEP 7: FINAL EVALUATION
# ============================================================================
    print("\n--- Step 7: Starting Final Evaluation of Adjusted Symbolic Model ---")

    model_final, eval_data_final, metrics_final = classic.evaluate_kan_model(
        model_save_path=final_model_prefix,
        X_test_tensor=X_test, y_test_tensor=y_test,
        conf_matrix_save_path=CONFIG["final_eval_cm"],
        save_path_roc_curve=CONFIG["final_eval_roc"],
        save_path_pr_curve=CONFIG["final_eval_pr"]
    )

    np.save(CONFIG["final_eval_data_true"], eval_data_final[0])
    np.save(CONFIG["final_eval_data_probs"], eval_data_final[1])
    np.save(CONFIG["final_eval_data_binary"], eval_data_final[2])

    
    
    end_time = time.time()
    pipeline_total_time = end_time - start_time

    metrics_final['total_pipeline_time_seconds'] = pipeline_total_time

    with open(CONFIG["final_eval_metrics"], 'w') as f:
        json.dump(metrics_final, f, indent=4)

    print(f"\n✅ Production pipeline completed in {end_time - start_time:.2f} seconds.")

    model_final.get_act(X_train[:10000])
    model_final.attribute()
    estructura_real = model_final.width[1]
    model_base.width[1] = [9, 0]

    print("\n--- Plotting final Model Splines ---")
    model_final.plot(
        folder=CONFIG["final_model_plot_folder"],
        save_path=CONFIG["final_model_plot_save_path"],
        beta=12.0,
        metric="backward",
        in_vars=CONFIG["features"],
        scale=1.0,
        varscale=0.5
    )

    clean_memory(model_final, eval_data_final, metrics_final,
                 X_test, y_test, X_sample, scaler)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated KAN Training Pipeline")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument("--force", action='store_true', help='Overwrite existing models')
    args = parser.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
