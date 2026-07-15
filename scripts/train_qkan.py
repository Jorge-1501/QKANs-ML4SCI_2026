# train_qkan.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import time
import os
import argparse
import copy
import json
import numpy as np

import sys
from pathlib import Path
# Añadir la raíz al path para importar src y workspace
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.architectures.qkan_model import plot_qkan_circuit
import src.preprocessing.processor_qg as processor
from src.architectures.qkan_model import QKANModel
from src.utils import workspace
import src.utils.metrics as viz

from src.utils.extract_weights import extract_chebyshev_weights
from src.utils.evaluate_qkan import evaluate_qkan

def train_quantum_model(args):
    torch.set_num_threads(4)  # Limit PyTorch to a single thread for reproducibility

    CONFIG = workspace.get_config(task="quark-gluon", seed=args.seed)
    workspace.make_dirs(CONFIG)

    train_backend = args.train_backend
    print(f"Selected backend mode: {train_backend} \n")

    # ============================================================================
    # Step 1: Extract Classical Knowledge (Chebyshev) an plotting circuit
    # ============================================================================
    extract_chebyshev_weights(CONFIG, force=args.force)

    plot_qkan_circuit(CONFIG["circuit_plot"])

    # ============================================================================
    # Step 2: Quantum Training
    # ============================================================================

    print("--- 1. Loading and Preprocessing Data ---")
    data_dir = os.path.join(CONFIG["raw_data_dir"], "quark-gluon")  # Assuming the data is in a subdirectory named 'quark-gluon'
    processed_data_dir = CONFIG["processed_data_dir"]
    # Structuring the data loading to get the original train/val/test splits
    X_train, y_train, X_val, y_val, _, _, _, _ = processor.load_and_preprocess_data(
                                                    data_dir=data_dir,
                                                    processed_dir=processed_data_dir,
                                                    task=args.task,
                                                    force_process=False)

    if train_backend == "noisy":
        save_path = CONFIG["qkan_noisy_path"]
        history_path = CONFIG["history_noisy_loss"]
        loss_plot_path = CONFIG["history_noisy_loss_plot"]
        auc_plot_path = CONFIG["history_noisy_auc_plot"]
    elif train_backend == "ideal":
        save_path = CONFIG["qkan_ideal_path"]
        history_path = CONFIG["history_ideal_loss"]
        loss_plot_path = CONFIG["history_ideal_loss_plot"]
        auc_plot_path = CONFIG["history_ideal_auc_plot"]
    elif train_backend == "shots":
        save_path = CONFIG["qkan_shots_path"]
        history_path = CONFIG["history_shots_loss"]
        loss_plot_path = CONFIG["history_shots_loss_plot"]
        auc_plot_path = CONFIG["history_shots_auc_plot"]

    if os.path.exists(save_path) and not args.force:
        print(f"Model, {train_backend}, already exists at '{save_path}'. Use '--force' to overwrite.")
        print(f"Loading existing model for evaluation...")
    else:
        batch_size = CONFIG["qkan_batch_size"]    
        n_val_samples = CONFIG["n_val_samples"]  # A fixed number of samples for validation
        torch.manual_seed(args.seed) # For reproducibility
        print(f"Validation set subsampled to {n_val_samples} data points for consistent evaluation")
        val_indices = torch.randperm(len(X_val))[:n_val_samples]
        X_val = X_val[val_indices]
        y_val = y_val[val_indices]

        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        # ==========================================================  

        print("\n--- 2. Initializing QKAN with Warm-Start ---")
        # We pass the path of .npy files
        model = QKANModel(init_weights_path=CONFIG["init_weights"])
        
        # Print the number of parameters for your thesis (should be 22: 4*5 + 2)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Quantum trainable parameters: {num_params}")

        # -------------------------------------------------------------
        # Fine-tuning with L2 regularization
        # -------------------------------------------------------------
        learning_rate = CONFIG["qkan_learning_rate"]
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)#, weight_decay=1e-4)  # L2 regularization with weight_decay
        criterion = nn.BCEWithLogitsLoss()
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, threshold=1e-3)

        # ------------------------------------------------------------
        # --- Checkpoint and early stop configuration ---
        # ------------------------------------------------------------

        num_epochs = CONFIG["qkan_epochs"]

        best_val_auc = 0.0
        best_val_loss = float('inf')
        best_model_weights = None

        patience = CONFIG["qkan_patience"]
        patience_counter = 0
        min_delta = CONFIG["qkan_early_stop_delta"]

        history = {"train_loss": [], "val_loss": [], "train_auc" : [], "val_auc": []}
        n_train_samples_for_epoch = CONFIG["n_train_samples_for_epoch"]  # Number of training samples to use per epoch (subsampling)
        # ============================================================
        
        starting_time_global = time.time()
        print("\n--- 3. Starting Quantum Fine-Tuning ---")
        print(f"\nExtracting a subsample of {n_train_samples_for_epoch} data points for quantum training...")
        for epoch in range(num_epochs):
            start_time = time.time()
            sampling_generator = torch.Generator().manual_seed(args.seed + 2*epoch)  # For reproducibility of subsampling
            # =========================================================
            # --- Diferent sample to train every epoch ---
            # =========================================================
            # Generate random indices (this ensures mix Quarks and Gluons)
            train_indices = torch.randperm(len(X_train), generator=sampling_generator)[:n_train_samples_for_epoch]
            
            # Apply the mask to the tensors
            X_train_sub = X_train[train_indices]
            y_train_sub = y_train[train_indices]

            train_dataset = torch.utils.data.TensorDataset(X_train_sub, y_train_sub)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            # =========================================================
            
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                # Important: batch_y must have the same shape (batch_size,) or (batch_size, 1)
                loss = criterion(outputs.squeeze(), batch_y.squeeze()) 
                if torch.isnan(loss):
                    print("Warning: NaN loss encountered. Skipping this batch.")
                    continue
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= len(train_dataset)

            # Validation
            model.eval()
            val_loss = 0.0
            all_val_probs = []
            all_val_true = []
            
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    outputs_val = model(batch_X_val)
                    loss_val = criterion(outputs_val.squeeze(), batch_y_val.squeeze())
                    val_loss += loss_val.item() * batch_X_val.size(0)
                    
                    probs = torch.sigmoid(outputs_val).numpy()
                    all_val_probs.extend(probs)
                    all_val_true.extend(batch_y_val.numpy())
                    
            val_loss /= len(val_dataset)
            if np.isnan(all_val_probs).any():
                print("Warning: NaN values in validation probabilities. Skipping AUC calculation for this epoch.")
                val_auc = 0.5
            else:
                val_auc = roc_auc_score(all_val_true, all_val_probs)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            current_lr = optimizer.param_groups[0]['lr']
            lr_scheduler.step(val_loss)

            epoch_time = time.time() - start_time
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] | Time: {epoch_time:.1f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # if lr changes, print it
            if epoch > 0 and current_lr != optimizer.param_groups[0]['lr']:
                print(f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f} at epoch {epoch+1} due to plateau in validation loss.")

            # Checkpointing based on validation AUC
            if val_auc > best_val_auc + min_delta:
                best_val_auc = val_auc
                best_model_weights = copy.deepcopy(model.state_dict())
                torch.save(best_model_weights, save_path)
                print(f"New best model saved with Val AUC: {best_val_auc:.4f} at epoch {epoch+1}.")

            # Early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs. \n")
                    break

        end_time_global = time.time()
        total_training_time = end_time_global - starting_time_global
        print(f"\nTotal training time: {total_training_time:.1f} seconds")

                # saving hiperparameters and metadata
        qkan_metadata = {
            "model_type": "Quantum Kernel Ansatz (QKAN)",
            "input_features": CONFIG["features"],
            "num_params": num_params,
            "optimizer": "Adam",
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "training_backend": train_backend,
            "evaluation_backend": args.eval_backend,
            "num_epochs": num_epochs,
            "early_stopping_patience": patience,
            "early_stopping_min_delta": min_delta,
            "n_train_samples_for_epoch": n_train_samples_for_epoch,
            "training_time_seconds": total_training_time
        }
        with open(CONFIG["qkan_metadata_path"], "w") as f:
            json.dump(qkan_metadata, f, indent=4)

        print(f"\n--- 4. Saving the Quantum Model in '{save_path}' ---")

        with open(history_path, 'w') as f:
            json.dump(history, f)

        # history graphics
        viz.plot_loss_history(history={"train_loss": history['train_loss'],
                                "val_loss": history['val_loss']},
                                save_path=CONFIG['history_noisy_loss_plot'])
                                
        viz.plot_auc_history(history=history, save_path=CONFIG['history_noisy_auc_plot'])

    # Evaluation
    evaluate_qkan(CONFIG, train_backend=args.train_backend, eval_backend=args.eval_backend)

# ===========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the QKAN quantum model with warm-started weights.")
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--train_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='ideal', help='Quantum backend mode to use for training')
    parser.add_argument('--eval_backend', type=str, choices=['noisy', 'ideal', 'shots'], default='noisy', help='Quantum backend mode to use for evaluation')
    parser.add_argument('--force', action='store_true', help='Force retraining even if model already exists')
    parser.add_argument('--task', type=str, choices=['top', 'quark-gluon'], default='quark-gluon', help='Task to perform: "top" for top quark classification or "quark-gluon" for quark-gluon discrimination')
    args = parser.parse_args()
    
    try:
        train_quantum_model(args)
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()