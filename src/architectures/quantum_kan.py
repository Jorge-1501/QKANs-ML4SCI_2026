# src/architectures/quantum_kan.py

import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.architectures.qkan_model import QKANModel

class QuantumKANTrainer:
    def __init__(self, config, train_backend="ideal"):
        """
        """
        self.config = config
        self.train_backend = train_backend
        
        # Set threads for PyTorch to avoid oversubscription in multi-threaded environments
        torch.set_num_threads(4)
        
        # Initialize the quantum model with Chebyshev Warm-Start
        self.model = QKANModel(init_weights_path=self.config["init_weights"])
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config["qkan_learning_rate"])
        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=6, threshold=1e-3
        )
        
        # Dynamic definition of paths according to the assigned quantum backend
        self._setup_backend_paths()

    def _setup_backend_paths(self):
        """Assign the correct paths from the CONFIG dictionary according to the backend."""
        if self.train_backend == "noisy":
            self.save_path = self.config["qkan_noisy_path"]
            self.history_path = self.config["history_noisy_loss"]
        elif self.train_backend == "ideal":
            self.save_path = self.config["qkan_ideal_path"]
            self.history_path = self.config["history_ideal_loss"]
        elif self.train_backend == "shots":
            self.save_path = self.config["qkan_shots_path"]
            self.history_path = self.config["history_shots_loss"]

    def fit(self, X_train, y_train, X_val, y_val, resume=True, force=False):
        """
        Runs the quantum training loop with fault tolerance per backend.
        Manages reproducible subsampling per epoch for HEP.
        """
        # Check if a checkpoint exists and handle resuming or forcing a new training session
        if os.path.exists(self.save_path) and resume and not force:
            print(f"\n[Q-Trainer] Quantum model ({self.train_backend}) detected at '{self.save_path}'. Loading...")
            self.model.load_state_dict(torch.load(self.save_path))
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f: return json.load(f)
            return {"train_loss": [], "val_loss": [], "val_auc": []}

        print(f"\n[Q-Trainer] Starting quantum fine-tuning in mode: {self.train_backend}")
        
        # Fixed subsampling for consistent validation
        n_val = self.config["n_val_samples"]
        val_indices = torch.randperm(len(X_val))[:n_val]
        X_val_sub, y_val_sub = X_val[val_indices], y_val[val_indices]
        
        val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_val_sub, y_val_sub), 
            batch_size=self.config["qkan_batch_size"], shuffle=False
        )

        best_val_auc = 0.0
        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_auc": []}
        
        starting_time = time.time()

        for epoch in range(self.config["qkan_epochs"]):
            epoch_start = time.time()
            
            # Dynamic and reproducible subsampling per epoch for HEP
            sampling_gen = torch.Generator().manual_seed(self.config["seed"] + 2 * epoch)
            train_indices = torch.randperm(len(X_train), generator=sampling_gen)[:self.config["n_train_samples_for_epoch"]]
            
            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train[train_indices], y_train[train_indices]),
                batch_size=self.config["qkan_batch_size"], shuffle=True
            )

            # --- Quantum training (Train) ---
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y.squeeze())
                
                if torch.isnan(loss): continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item() * batch_X.size(0)
                
            train_loss /= self.config["n_train_samples_for_epoch"]

            # --- Quantum evaluation (Validation) ---
            self.model.eval()
            val_loss = 0.0
            all_val_probs, all_val_true = [], []
            
            with torch.no_grad():
                for batch_X_val, batch_y_val in val_loader:
                    outputs_val = self.model(batch_X_val)
                    loss_val = self.criterion(outputs_val.squeeze(), batch_y_val.squeeze())
                    val_loss += loss_val.item() * batch_X_val.size(0)
                    
                    all_val_probs.extend(torch.sigmoid(outputs_val).numpy())
                    all_val_true.extend(batch_y_val.numpy())
                    
            val_loss /= n_val
            val_auc = 0.5 if np.isnan(all_val_probs).any() else roc_auc_score(all_val_true, all_val_probs)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_auc"].append(val_auc)

            self.lr_scheduler.step(val_loss)
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{self.config['qkan_epochs']}] | Loss Train: {train_loss:.4f} | Loss Val: {val_loss:.4f} | AUC Val: {val_auc:.4f}")

            # Checkpointing based on quantum AUC
            if val_auc > best_val_auc + self.config["qkan_early_stop_delta"]:
                best_val_auc = val_auc
                torch.save(copy.deepcopy(self.model.state_dict()), self.save_path)
                print(f" -> Saved best quantum model (AUC: {best_val_auc:.4f})")

            # Early stopping based on loss
            if val_loss < best_val_loss - self.config["qkan_early_stop_delta"]:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config["qkan_patience"]:
                    print(f" Early Stopping Quantum activated at epoch {epoch+1}.")
                    break

        # Save metadata and history at the end of the actual training
        with open(self.history_path, 'w') as f: json.dump(history, f)
        return history
