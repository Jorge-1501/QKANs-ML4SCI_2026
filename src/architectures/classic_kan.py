# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# classic_kan.py

# Import necessary packages
import time #, math
import warnings

#from scipy.stats import ks_2samp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, auc, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    precision_score, recall_score, roc_curve
)
from kan import KAN
import gc
import json
#import traceback
from kan.utils import SYMBOLIC_LIB
#import sympy
from sympy.utilities.lambdify import lambdify

import src.utils.metrics as viz



# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# ============================================================================
# Define the neural network model
# ============================================================================
# ============================================================================

def train_kan_model(width, grid, k, learning_rate, num_epochs, batch_size,
                           early_stop_patience, early_stop_min_delta, 
                           model_save_path,
                           X_train_tensor,
                           y_train_tensor,
                           X_val_tensor,
                           y_val_tensor,
                           lamb=0.01, 
                           lamb_l1=1.0, 
                           lamb_entropy=2.0,
                           lamb_coef=0.0,
                           lamb_coefdiff=0.0,
                           reg_metric='edge_forward_spline_n',
                           update_grid_freq=5,
                           num_workers=4):
    """
    Trains a KAN model using a custom loop that includes key pykan features:
    regularization and adaptive grid updates.

    Args:
        width (list): The architecture of the KAN model.
        grid (int): The number of grid points for the splines.
        k (int): The order of the splines.
        learning_rate (float): The learning rate for the Adam optimizer.
        num_epochs (int): The maximum number of epochs for training.
        batch_size (int): The size of the batches for training.
        early_stop_patience (int): Number of epochs with no improvement to wait before stopping.
        early_stop_min_delta (float): Minimum change in validation loss to be considered an improvement.
        model_save_path (str): Path to save the best model checkpoint.
        X_train_tensor, y_train_tensor: Training data and labels as PyTorch tensors.
        X_val_tensor, y_val_tensor: Validation data and labels as PyTorch tensors.
        lamb (float): Overall regularization strength.
        lamb_l1 (float): L1 regularization strength for sparsity.
        lamb_entropy (float): Entropy regularization strength.
        update_grid_freq (int): The frequency (in epochs) to update the spline grid.
        num_workers (int): Number of worker processes for data loading.
    Returns:
        dict: A dictionary containing the training history (losses and AUCs).
    """
    print(f"\n--- Starting Training for KAN Model ---")
    print(f"Parameters: width={width}, grid={grid}, k={k}, lr={learning_rate}")

    # 1. Define the model, criterion, and optimizer
    model = KAN(width, grid, k, symbolic_enabled=True, auto_save=False)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"KAN model created with {num_params} parameters.")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 2. Set up optimized DataLoaders
      # Use a reasonable number of workers based on CPU cores
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, persistent_workers=False)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, persistent_workers=False)
    
    # 3. Training and Validation Loop
    best_val_auc = 0.0
    epochs_no_improve = 0
    best_val_loss = float('inf')
    start_time = time.time()

    history = {'train_loss': [], 
               'val_loss': [],
               'train_auc': [],
                'val_auc': []
               }

    print(f"\nStarting training using {num_workers} workers...")
    for epoch in range(num_epochs):
        # --- Update the grid adaptively ---
        if epoch > 0 and epoch % update_grid_freq == 0:
            
            # New grid size
            new_grid_size = model.grid + 3
            
            # Creating a new model with the new grid size
            new_model = KAN(width, new_grid_size, k, symbolic_enabled=True, auto_save=False)
            
            # Transfer weights from the old model to the new model
            new_model.initialize_from_another_model(model, X_train_tensor)
            model = new_model
            
            # Adjust learning rate and early stopping delta for the grid update
            #learning_rate *= 0.8
            early_stop_min_delta = max(early_stop_min_delta * 0.6, 1e-6) # Reduce delta but not below a minimum threshold
            
            # Reset early stopping counters after grid update
            epochs_no_improve = 0
            best_val_loss = float('inf')

            print(f"\nUpdating grid at epoch {epoch+1}...")
            print(f"  -> Current grid: {model.grid}")
            print(f"learning_rate for grid update: {learning_rate:.6f}")
            print(f"early_stop_min_delta for grid update: {early_stop_min_delta:.6f}")
            
            print(f"Model grid updated to {model.grid} points. Continuing training with the new grid...")
            
            # Use the training set to inform the grid update
            with torch.no_grad():
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                #model.update_grid(X_train_tensor) 

        model.train()
        train_loss = 0.0

        all_train_probs_list = []
        all_train_true_list = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate the classification loss
            bce_loss = criterion(outputs, batch_y)
            
            # --- KAN Regularization Loss ---
            reg_value = model.get_reg(reg_metric, lamb_l1, lamb_entropy, lamb_coef, lamb_coefdiff)
            
            # Combine both losses
            total_loss = bce_loss + lamb*reg_value

            total_loss.backward()
            optimizer.step()
            # We only track the classification loss for logging purposes
            train_loss += bce_loss.item() * batch_X.size(0)

            all_train_probs_list.append(torch.sigmoid(outputs).cpu().detach())
            all_train_true_list.append(batch_y.cpu().detach())
            
        train_loss /= len(train_dataset)

        # Validation process
        model.eval()
        val_loss = 0.0

        all_val_probs_list = []
        all_val_true_list = []
        
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                val_loss += loss_val.item() * batch_X_val.size(0)

                all_val_probs_list.append(torch.sigmoid(outputs_val).cpu().detach())
                all_val_true_list.append(batch_y_val.cpu().detach())
        val_loss /= len(val_dataset)

        # Concatenate all predictions and true labels
        all_val_probs = torch.cat(all_val_probs_list, dim=0).numpy()
        all_val_true = torch.cat(all_val_true_list, dim=0).numpy()

        all_train_probs = torch.cat(all_train_probs_list, dim=0).numpy()
        all_train_true = torch.cat(all_train_true_list, dim=0).numpy()

        # AUC calculation for all validation data
        val_auc = roc_auc_score(all_val_true, all_val_probs)
        train_auc = roc_auc_score(all_train_true, all_train_probs)

        # logging for every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Training Loss: {train_loss:.5f}, "
                f"Validation Loss: {val_loss:.5f}, "
                f"Training AUC: {train_auc:.5f}, "
                f"Validation AUC: {val_auc:.5f}")

        # Save losses and AUC
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Check for improvement in validation AUC and val loss for early stopping and model checkpointing
        if ((val_auc > best_val_auc + early_stop_min_delta) \
        or (val_loss < best_val_loss - early_stop_min_delta)):
            best_val_auc = val_auc
            best_val_loss = val_loss
            epochs_no_improve = 0
            model.saveckpt(model_save_path)

            training_time_seconds = time.time() - start_time
            metadata = {
                'num_params': num_params,
                'training_time_seconds': training_time_seconds,
                'final_val_auc': best_val_auc,
                'final_val_loss': best_val_loss,
                'hyperparameters': {
                    'width': width,
                    'grid': grid,
                    'k': k,
                    'learning_rate': learning_rate
                }
            }
            try:
                with open(f"{model_save_path}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
            except Exception as e:
                print(f"Warning: Could not save metadata to JSON file. Error: {e}")
            print(f"Epoch [{epoch+1}/{num_epochs}] - Model checkpoint saved (val_auc: {best_val_auc:.5f} - val_loss: {val_loss:.5f}).")    
        else:
            epochs_no_improve += 1

        # Early stop based on loss improvement
        if val_loss < best_val_loss - early_stop_min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print("Early Stopping triggered!")
            break
            
    training_time_seconds = time.time() - start_time
    print(f"\nTraining finished in {training_time_seconds:.2f} seconds!")
    
    # --- Save a dictionary with model state and metadata ---
    print(f"Saving final model and metadata to '{model_save_path}'...")

    return history

def clean_memory(*args):
    """Utility function to clean up memory by deleting variables and clearing GPU cache."""
    for obj in args:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def evaluate_kan_model(model_save_path, 
                       X_test_tensor, y_test_tensor, 
                       save_path_roc_curve=None, 
                       conf_matrix_save_path=None,
                       save_path_pr_curve=None):
    """
    Loads a KAN model and its metadata from a checkpoint file and evaluates 
    its performance on the test set.
    args:
        model_save_path (str): Path to the saved model checkpoint (.pth file).
        X_test_tensor (torch.Tensor): Test features as a PyTorch tensor.
        y_test_tensor (torch.Tensor): Test labels as a PyTorch tensor.
        save_path_roc_curve (str): Path to save the ROC curve plot.
        conf_matrix_save_path (str): Path to save the confusion matrix plot.
        save_path_pr_curve (str): Path to save the Precision-Recall curve plot.
    return:
        tuple: A tuple containing the loaded model, 
               a tuple of (true labels, predicted probabilities, predicted classes),
               and a dictionary of evaluation metrics.
    """
    print("\n--- Starting Evaluation of the Best Model ---")

    model = KAN.loadckpt(model_save_path)
    
    # 3. Load the trained weights into the model
    model.eval()
    print(f"Model loaded from '{model_save_path}'.")

    # 5. Evaluate on the test set (The rest of the function is the same)
    criterion = nn.BCEWithLogitsLoss()
    test_true = y_test_tensor.cpu().numpy()

    with torch.no_grad():
        outputs_test = model(X_test_tensor)
        loss_test = criterion(outputs_test, y_test_tensor)
        
        probs_test = torch.sigmoid(outputs_test)
        predicted_classes_test = (probs_test > 0.5).float()
        
        test_preds_probs = probs_test.cpu().numpy()
        test_preds_binary = predicted_classes_test.cpu().numpy()

    # 3. Calculate and display final metrics
    test_loss = loss_test.item()
    test_accuracy = accuracy_score(test_true, test_preds_binary)
    test_f1 = f1_score(test_true, test_preds_binary)
    test_auc = roc_auc_score(test_true, test_preds_probs)
    test_precision = precision_score(test_true, test_preds_binary)
    test_recall = recall_score(test_true, test_preds_binary)
    conf_matrix = confusion_matrix(test_true, test_preds_binary)
        
    # --- Print Final Metrics ---
    # Now using the metadata loaded from the file
    print("\n--- Final Metrics on the Test Set ---")
    print(f"Test Loss: {test_loss:.5f}")
    print(f"Test Accuracy: {test_accuracy:.5f}")
    print(f"Test F1 Score: {test_f1:.5f}")
    print(f"Test AUC: {test_auc:.5f}")
    print(f"Test Precision: {test_precision:.5f}")
    print(f"Test Recall: {test_recall:.5f}")

    # Print Confusion Matrix
    print("\nMatriz de Confusión:")
    print(conf_matrix)

    # plot ROC curve
    viz.plot_roc_curve(test_true, test_preds_probs, save_path=save_path_roc_curve)

    # plot Precision-Recall curve
    viz.plot_precision_recall_curve(test_true, test_preds_probs, save_path=save_path_pr_curve)

    # Plot Confusion Matrix
    viz.plot_confusion_matrix(conf_matrix, save_path=conf_matrix_save_path)

    metrics = {
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy,
        "Test F1 Score": test_f1,
        "Test AUC": test_auc,
        "Test Precision": test_precision,
        "Test Recall": test_recall,
        "Confusion Matrix": conf_matrix.tolist() # .tolist() to make it JSON serializable
    }

    return model, (test_true, test_preds_probs, test_preds_binary), metrics

# ============================================================================
# Pruning and Retraining Functions
# ============================================================================

def prune_and_save_kan(original_model_path,
                        pruned_model_path,
                        activation_data,
                        node_th=1e-2,
                        edge_th=3e-2
                       ):
    """
    Load a trained KAN model, prune it, and save a new checkpoint
    compatible with the evaluation function.

    Args:
        original_model_path (str): Path to the .pth file of the trained model.
        pruned_model_path (str): Path where the new checkpoint of the pruned model will be saved.
        prune_threshold (float): Threshold for pruning nodes and edges.

    Returns:
        KAN: The pruned model object, ready for retraining.
    """
    print(f"\n--- Starting Pruning Process ---")

    # 1. Load the original complete checkpoint
    print(f"Loading original model from (prefix): '{original_model_path}'")
    original_model = KAN.loadckpt(original_model_path)

        # 2. Prune the model
    print(f"Pruning the model with a threshold of {node_th} and {edge_th}...")
    # The model needs data to compute activations before pruning
    # We assume training data is available (you can pass X_train_tensor if not)
    original_model.get_act(activation_data) 
    pruned_model = original_model.prune(node_th=node_th, edge_th=edge_th)
    print(f"New architecture after pruning: {pruned_model.width}")

    # 3. Create a new compatible checkpoint for the pruned model
    print(f"Creating and saving the new checkpoint at: '{pruned_model_path}'")

    pruned_model.saveckpt(pruned_model_path)
    
    print("Checkpoint of the pruned model saved successfully.")
    
    return pruned_model

# retrain function
def retrain_pruned_kan(pruned_model,
                        learning_rate,
                        num_epochs, batch_size,
                        early_stop_patience,
                        early_stop_min_delta,
                        model_save_path,
                        X_train_tensor,
                        y_train_tensor,
                        X_val_tensor,
                        y_val_tensor,
                        lamb=0.0,
                        lamb_l1=0.0,
                        lamb_entropy=0.0, 
                        lamb_coef=0.0,
                        lamb_coefdiff=0.0,
                        num_workers = 1
                         ):
    """
    Retrain (fine-tuning) a KAN model pruned and save the best model.
    The pruned_model should be an instance of KAN already pruned.

    Args:
        pruned_model (KAN): The pruned KAN model instance.
        learning_rate (float): Learning rate for the Adam optimizer.
        num_epochs (int): Maximum number of epochs for retraining.
        batch_size (int): Batch size for training.
        early_stop_patience (int): Number of epochs with no improvement to wait before stopping.
        early_stop_min_delta (float): Minimum change in validation loss to be considered an improvement.
        model_save_path (str): Path to save the best retrained model.
        X_train_tensor, y_train_tensor: Training data and labels as PyTorch tensors.
        X_val_tensor, y_val_tensor: Validation data and labels as PyTorch tensors.
        lamb (float): Overall regularization strength for retraining.
        lamb_l1 (float): L1 regularization strength for sparsity during retraining.
        lamb_entropy (float): Entropy regularization strength for retraining.
        lamb_coef (float): Coefficient regularization strength for retraining.
        lamb_coefdiff (float): Coefficient difference regularization strength for retraining.
        num_workers (int): Number of worker processes for data loading.
        
    """
    print(f"\n--- Start Retraining (Fine-Tuning) the Pruned Model ---")
    print(f"Using a learning rate of: {learning_rate}")

    # 1. The model already exists, we just need the criterion and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(pruned_model.parameters(), lr=learning_rate)

    # 2. Configure DataLoaders (same as before)
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=num_workers, persistent_workers=False)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             num_workers=num_workers, persistent_workers=False)

    # 3. Retraining Loop
    best_val_auc = float('-inf')
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_time = time.time()

    history = {'train_loss': [], 
               'val_loss': [], 
               'train_auc': [], 
               'val_auc': []}

    print(f"\nStarting Fine-Tuning...")
    for epoch in range(num_epochs):
        pruned_model.train()
        train_loss = 0.0

        all_train_probs_list = []
        all_train_true_list = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = pruned_model(batch_X)
            loss = criterion(outputs, batch_y)
            reg_value = pruned_model.get_reg('edge_forward_spline_n', 
                                            lamb_l1, lamb_entropy, 
                                            lamb_coef, lamb_coefdiff)
            loss = loss + lamb * reg_value
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            all_train_probs_list.append(torch.sigmoid(outputs).cpu().detach())
            all_train_true_list.append(batch_y.cpu().detach())
        train_loss /= len(train_dataset)

        # Complete validation process
        pruned_model.eval()
        val_loss = 0.0

        all_val_probs_list = []
        all_val_true_list = []

        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = pruned_model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                val_loss += loss_val.item() * batch_X_val.size(0)
                all_val_probs_list.append(torch.sigmoid(outputs_val).cpu().detach())
                all_val_true_list.append(batch_y_val.cpu().detach())
        val_loss /= len(val_dataset)

        # Concatenate all predictions and true labels
        all_val_probs = torch.cat(all_val_probs_list, dim=0).numpy()
        all_val_true = torch.cat(all_val_true_list, dim=0).numpy()

        all_train_probs = torch.cat(all_train_probs_list, dim=0).numpy()
        all_train_true = torch.cat(all_train_true_list, dim=0).numpy()

        # AUC calculation for all validation data
        val_auc = roc_auc_score(all_val_true, all_val_probs)
        train_auc = roc_auc_score(all_train_true, all_train_probs)

        print(f"Retraining Epoch [{epoch+1}/{num_epochs}], "
              f"Training Loss: {train_loss:.5f}, "
              f"Validation Loss: {val_loss:.5f}, "
              f"Training AUC: {train_auc:.5f}, "
              f"Validation AUC: {val_auc:.5f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        # Early Stopping logic and saving the best retrained model
        # base on loss and auc improvement
        if (val_loss < best_val_loss - early_stop_min_delta) \
            or (val_auc > best_val_auc + early_stop_min_delta):
            best_val_loss = val_loss
            best_val_auc = val_auc
            epochs_no_improve = 0

            pruned_model.saveckpt(model_save_path)
            print(f"Best retrained model saved at epoch {epoch+1} with val loss: {best_val_loss:.5f}")

            # --- Save the complete dictionary ---
            retraining_time = time.time() - start_time
            num_params_pruned = sum(p.numel() for p in pruned_model.parameters() if p.requires_grad)

            metadata = {
                'num_params': num_params_pruned,
                'training_time_seconds': retraining_time, # Save retraining time
                'final_val_loss': best_val_loss,
                'final_val_auc': best_val_auc,
                'hyperparameters': {
                    'width': pruned_model.width, # New pruned architecture
                    'grid': pruned_model.grid,   # Original grid
                    'k': pruned_model.k,          # Original k
                    'learning_rate': learning_rate
                }
            }
            try:
                with open(f"{model_save_path}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
            except Exception as e:
                print(f"Warning: Could not save metadata to JSON file. Error: {e}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stop_patience:
            print("¡Early Stopping activated for retraining!")
            break
            
    retraining_time_total = time.time() - start_time
    print(f"\nRetraining completed in {retraining_time_total:.2f} seconds!")

    # Load the best saved model to return it
    final_model = KAN.loadckpt(model_save_path)

    return final_model, history

# ============================================================================
# Automatized Symbolic Fitting
# ============================================================================

# --- Step 1: Global Search ---
def find_best_symbolic_globally(model, all_function_names):
    '''
    Performs a global search over all edges and all functions in the SYMBOLIC_LIB.
    Logs the R² and complexity for each function tested on each edge.
    Arg:
        model: KAN model and pruned (KAN instance).
        all_function_names: List of all function names in SYMBOLIC_LIB.
        weight_simple: Weighting factor for complexity vs R2 in cost function.
    Return:
        edge_results: Dictionary with results for each edge.
    '''
    print("Starting global search. Thank you for your patience.")
    # This dictionary will store the results: (l,i,j) -> [ (name, r2, complexity), ... ]
    edge_results = {}
    
    # 1. Iterate through all the edges
    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l + 1]):
                
                edge = (l,i,j)

                # 2. Skip edges that are already fixed or masked (same as auto_symbolic)
                if model.symbolic_fun[l].mask[j, i] > 0. and model.act_fun[l].mask[i][j] == 0.:
                    print(f"Skipping {edge}: already symbolic.")
                    continue
                elif model.symbolic_fun[l].mask[j, i] == 0. and model.act_fun[l].mask[i][j] == 0.:
                    print(f"Skipping {edge}: associated edge (value 0).")
                    continue
                
                print(f"--- Testing edge {edge} ---")
                edge_results[edge] = []

                # 3. Testing each function from the complete library
                # (This is slow, this is where grid_number matters)
                for fun_name in all_function_names:
                    try:
                        # 4. Using fix_symbolic/unfix_symbolic to *test*
                        # (This is what suggest_symbolic does internally)
                        r2 = model.fix_symbolic(l, i, j, fun_name, verbose=False, log_history=False)

                        if r2 != -1e8: # If the function is not a zero function (masked)
                            # Get the complexity
                            c = SYMBOLIC_LIB[fun_name][2]
                            edge_results[edge].append((fun_name, r2.item(), c))
                            gc.collect() # Free memory

                        # 5. Undo the change to test the next function
                        model.unfix_symbolic(l, i, j, log_history=False)

                    except Exception as e:
                        # If a function fails (e.g. IndexError), we log it and continue
                        print(f"  Error testing {fun_name} on {edge}: {type(e).__name__}")
                        model.unfix_symbolic(l, i, j, log_history=False) # Ensure reset
                        gc.collect() # Free memory
                        continue
    
    print("Step 1 completed.")
    return edge_results

# --- STEP 2: ANALYSIS AND ADJUSTMENT (Make decisions) ---
def apply_best_symbolic_from_log(model, edge_results, r2_threshold=0.85, weight_simple=0.4):
    '''
    Reviews the results log and fixes the BEST function for each edge.
    Args:
        model: KAN model to modify (KAN instance).
        edge_results: Dictionary with results for each edge.
        r2_threshold: Minimum R² to consider a function acceptable.
        weight_simple: Weighting factor for complexity vs R2 in cost function.
    '''
    print("\nStarting Step 2: Analysis and Adjustment.")
    fix_log = []

    # Cost functions (identical to those in suggest_symbolic)
    r2_loss_fun = lambda x: np.log2(1+1e-5-x)
    c_loss_fun = lambda x: x

    for edge, results in edge_results.items():
        if not results:
            print(f"SKIPPING {edge}: No valid results.")
            continue # Edge without valid results

        # 1. First, create a list of only the functions that meet the R² threshold.
        candidate_functions = []
        for (name, r2, c) in results:
            if r2 >= r2_threshold:
                # Calculate the cost ONLY for functions that are good enough
                r2_loss = r2_loss_fun(r2)
                c_loss = c_loss_fun(c)
                total_loss = weight_simple * c_loss + (1 - weight_simple) * r2_loss
                candidate_functions.append((name, r2, c, total_loss))
        
        # 2. If no function met the threshold, skip this edge.
        if not candidate_functions:
            print(f"SKIPPING {edge}: No function met the R² threshold of {r2_threshold}")
            continue

        # 3. From the candidates, find the one with the lowest total cost (the best balance).
        candidate_functions.sort(key=lambda x: x[3]) # Sort by total_loss
        best_fun_name, best_r2, best_c, best_loss = candidate_functions[0]

        # 4. FIX the best function from the candidates
        print(f"ADJUSTING {edge}: {best_fun_name} (R2={best_r2:.5f}, C={best_c}, Loss={best_loss:.5f})")
        model.fix_symbolic(edge[0], edge[1], edge[2], best_fun_name, verbose=False, log_history=True)
        fix_log.append(f"Edge {edge}: {best_fun_name} (R2={best_r2:.5f})")

    print("\nStep 2 (Adjustment) completed.")

    return fix_log

# ============================================================================
# Final finetuning after symbolic fitting
# ============================================================================

def simplify_and_save(source_checkpoint_path, symbolic_model_path, 
                             x_train_sample,
                             r2_threshold=0.8, weight_simple=0.5):
    """
    Simplifies a KAN model by fitting symbolic functions to its edges
    based on R² threshold and complexity weight. Saves the final symbolic
    model to the specified path.
    Args:
        source_checkpoint_path (str): Path to the checkpoint from which the model will be loaded.
        symbolic_model_path (str): Path where the new checkpoint for the symbolic model will be saved.
        x_train_sample (torch.Tensor): Sample of the training set for calculating activations.
        r2_threshold (float): R² threshold for accepting a symbolic function.
        weight_simple (float): Weight given to simplicity vs. R² fit.
    Returns:
        KAN: The final model object with symbolic adjustments applied.
    """
    print("--- Starting Simplification Process ---")
    
    # 1. Load the model from the checkpoint
    print(f"Loading model from: '{source_checkpoint_path}'")
    start_time = time.time()
    model = KAN.loadckpt(source_checkpoint_path)
    model.eval()

    # Calculate and store activations before doing anything else.
    print("Calculating model activations...")
    model.get_act(x_train_sample)

    fix_log = []
    all_function_names = list(SYMBOLIC_LIB.keys())

    # Cost functions
    r2_loss_fun = lambda x: np.log2(1 + 1e-5 - x)
    c_loss_fun = lambda x: x

    for l in range(len(model.width_in) - 1):
        for i in range(model.width_in[l]):
            for j in range(model.width_out[l+1]):
                edge = (l, i, j)

                # If mask = 0, conexion doesn´t exist  
                if model.act_fun[l].mask[i][j] == 0:
                    continue

                print(f"\n--- Analyzing Edge {edge} ---")
                edge_results = []

                for fun_name in all_function_names:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                    try:
                        # fit_params_bool=True try to fit a*fun(b*x+c)+d
                        r2 = model.fix_symbolic(l, i, j, fun_name, fit_params_bool=True, verbose=False, log_history=False)

                        # Verify if r2 is a number
                        if r2 is not None and not np.isnan(r2.item()):
                            r2_val = r2.item()
                            # Filter values
                            if r2_val > -1e7:
                                print(f" -> Trying '{fun_name}': R2={r2.item():.5f}")
                                c = SYMBOLIC_LIB[fun_name][2]
                                total_loss = weight_simple * c_loss_fun(c) + (1 - weight_simple) * r2_loss_fun(r2.item())
                                edge_results.append((fun_name, r2.item(), c, total_loss))
                            else:
                                pass
                        # unfix to try next function
                        model.unfix_symbolic(l, i, j, log_history=False)
                    except Exception as e:
                        print(f"  -> Error while trying '{fun_name}': {type(e).__name__}: {e}")
                        model.unfix_symbolic(l, i, j, log_history=False)
                        continue
                
                # Selection process
                if not edge_results:
                    print(f"  -> Valid functions not found for this edge. Skipping.")
                    continue
                    
                # Ordering with loss increasing
                edge_results.sort(key=lambda x: x[3])
                best_fun_name, best_r2, best_c, best_loss = edge_results[0]

                if best_r2 >= r2_threshold:
                    print(f"  -> Fitting '{best_fun_name}' (R2={best_r2:.5f})")
                    # Applying change
                    model.fix_symbolic(l, i, j, best_fun_name, fit_params_bool=True, verbose=False, log_history=True)
                    fix_log.append(f"Edge {edge}: {best_fun_name} (R2={best_r2:.5f})")
                else:
                    print(f"  -> Skipping. Best R2={best_r2:.5f} < {r2_threshold}. Adjusting to identity, ('x'), by default.")
                    # Fallback to indentity 
                    model.fix_symbolic(l, i, j, 'x', fit_params_bool=True, verbose=False, log_history=True)
                    fix_log.append(f"Edge {edge}: 'x' (fallback, R2={best_r2:.5f})")
                # Cleaning memory
                gc.collect()

    print("\n--- Finalizing ---")
    # Saving the final symbolic model
    print(f"Saving final symbolic model to: '{symbolic_model_path}'")

    model.saveckpt(symbolic_model_path)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    simplification_time = time.time() - start_time
    print(f"Simplification completed in {simplification_time:.2f} seconds.")
    meta_data={
        'num_params': num_params,
        'simplification_time_seconds': simplification_time,
        'r2_threshold': r2_threshold,
        'weight_simple': weight_simple,
        'hyperparameters': {
                    'width': model.width,
                    'grid': model.grid,
                    'k': model.k,
                }
    }

    try:
        with open(f"{symbolic_model_path}_metadata.json", "w") as f:
            json.dump(meta_data, f, indent=4)
    except Exception as e:
        print(f"Warning: Could not save simplification metadata to JSON file. Error: {e}")

    print("\n" + "=" * 40 + "\nSummary of Fixes:")
    for line in fix_log: print(line)
    print("=" * 40)

    return model

def finetune_symbolic_model(
    source_symbolic_path, 
    final_model_path,
    X_train_tensor,
    y_train_tensor,
    X_val_tensor,
    y_val_tensor,
    learning_rate=1e-3,
    num_epochs=20,
    batch_size=64,
    early_stop_patience=5,
    early_stop_min_delta=5e-4,
    num_workers = 1
):
    """
    Fine-tune only the last layer of a symbolic KAN model and save the best model.
    Args:
        source_symbolic_path (str): Path to the .pth file of the symbolic model.
        final_model_path (str): Path where the final fine-tuned model will be saved.
        X_train_tensor, y_train_tensor: Training data and labels as PyTorch tensors.
        X_val_tensor, y_val_tensor: Validation data and labels as PyTorch tensors.
        learning_rate (float): Learning rate for the Adam optimizer.
        num_epochs (int): Maximum number of epochs for fine-tuning.
        batch_size (int): Batch size for training.
        early_stop_patience (int): Number of epochs with no improvement to wait before stopping.
        early_stop_min_delta (float): Minimum change in validation loss to be considered an improvement.
    Returns:
        KAN: The fine-tuned model object.
    """
    print("\n--- Starting Fine-Tuning ---")

    # 1. Load the symbolic model
    print(f"Loading symbolic model from: (prefix) '{source_symbolic_path}'")
    model = KAN.loadckpt(source_symbolic_path)
    """checkpoint = torch.load(source_symbolic_path)
    hp = checkpoint['hyperparameters']
    model = KAN(width=hp['width'], grid=hp['grid'], k=hp['k'], symbolic_enabled=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    """
    # 2. Freeze all layers except the last symbolic layer
    for param in model.parameters():
        param.requires_grad = False
    last_layer_index = len(model.symbolic_fun) - 1
    for param in model.symbolic_fun[last_layer_index].parameters():
        param.requires_grad = True
        
    # Bias and scale of the last layer (if they exist)
    if model.node_bias and last_layer_index < len(model.node_bias):
        model.node_bias[last_layer_index].requires_grad = True

    if model.node_scale and last_layer_index < len(model.node_scale):
        model.node_scale[last_layer_index].requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # 3. Configure DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    best_val_auc = float('-inf')
    epochs_no_improve = 0
    start_time = time.time()

    history = {
        'train_loss': [], 
        'val_loss': [],
        'train_auc': [],
        'val_auc': []
    }

    print(f"Fitting the last symbolic layer for a max of {num_epochs} epochs...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        all_train_probs_list = []
        all_train_true_list = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            all_train_probs_list.append(torch.sigmoid(outputs).cpu().detach())
            all_train_true_list.append(batch_y.cpu().detach())
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0

        all_val_probs_list = []
        all_val_true_list = []
        
        with torch.no_grad():
            for batch_X_val, batch_y_val in val_loader:
                outputs_val = model(batch_X_val)
                loss_val = criterion(outputs_val, batch_y_val)
                val_loss += loss_val.item() * batch_X_val.size(0)
                all_val_probs_list.append(torch.sigmoid(outputs_val).cpu().detach())
                all_val_true_list.append(batch_y_val.cpu().detach())
        val_loss /= len(val_dataset)

        # Concatenate all predictions and true labels
        all_val_probs = torch.cat(all_val_probs_list, dim=0).numpy()
        all_val_true = torch.cat(all_val_true_list, dim=0).numpy()
        all_train_probs = torch.cat(all_train_probs_list, dim=0).numpy()
        all_train_true = torch.cat(all_train_true_list, dim=0).numpy()

        # AUC calculation for all validation data
        val_auc = roc_auc_score(all_val_true, all_val_probs)
        train_auc = roc_auc_score(all_train_true, all_train_probs)

        print(f"Fitting epoch [{epoch+1}/{num_epochs}]", 
              f"Training Loss: {train_loss:.5f}", 
              f"Validation Loss: {val_loss:.5f}",
              f"Training AUC: {train_auc:.5f}", 
              f"Validation AUC: {val_auc:.5f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_auc'].append(train_auc)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc + early_stop_min_delta:
            best_val_auc = val_auc
            epochs_no_improve = 0

            # save the best model
            model.saveckpt(final_model_path)
            finetune_time = time.time() - start_time
            
            metadata = {
                'num_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
                'finetuning_time_seconds': finetune_time,
                'hyperparameters': {
                    'width': model.width, # New pruned architecture
                    'grid': model.grid,   # Original grid
                    'k': model.k,          # Original k
                    'learning_rate': learning_rate
                }
            }
        
            try:
                with open(f"{final_model_path}_metadata.json", "w") as f:
                    json.dump(metadata, f, indent=4)
            except Exception as e:
                print(f"Warning: Could not save finetuning metadata to JSON file. Error: {e}")
            
            print(f"  -> Fitting model saved: (Val AUC: {best_val_auc:.5f}) - (Val Loss: {val_loss:.5f})")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stop_patience:
            print("Early Stopping activated during fine-tuning!")
            break

    # Load the best saved model
    print(f"Loading the best model from '{final_model_path}'")
    best_model = KAN.loadckpt(final_model_path)
    #model.load_state_dict(best_model['model_state_dict'])

    # Unfreeze all layers at the end
    for param in best_model.parameters():
        param.requires_grad = True

    return best_model, history

def evaluate_symbolic_formula(formula, variables, X_test_tensor, y_test_tensor):
    """
    Evaluates a symbolic formula on the test set and computes its AUC.
    Args:
        formula: The symbolic formula to evaluate.
        variables: The variables used in the formula.
        X_test_tensor: The input features for the test set.
        y_test_tensor: The true labels for the test set.
    Returns:
        None: Prints the AUC of the symbolic formula.
    """
    print("Transforming symbolic formula to fast numerical function...")

    # 1. Transform the SymPy formula into a numerical function that NumPy can execute
    numeric_formula = lambdify(variables, formula, 'numpy')

    # 2. Prepare the data: Convert tensors to NumPy arrays
    X_test_np = X_test_tensor.cpu().numpy()
    y_test_np = y_test_tensor.cpu().numpy().flatten()

    # 3. Passing the 4 feature arrays (columns) to the function
    #    X_test_np[:, 0] is 'pT_log_norm', X_test_np[:, 1] is 'eta_norm', etc.
    try:
        logits = numeric_formula(
            X_test_np[:, 0],
            X_test_np[:, 1],
            X_test_np[:, 2],
            X_test_np[:, 3]
        )
    except Exception as e:
        print(f"Error evaluating numerical formula. Are the variables in the correct order? Error: {e}")
        return None

    # 4. Transform logits to probabilities (same as in the model)
    probs = 1 / (1 + np.exp(-logits)) 
    preds_binary = (probs > 0.5).astype(int)
    
    # 5. Compute AUC
    formula_auc = roc_auc_score(y_test_np, probs)

    metrics = {
        "Test Loss": -1, # skipping 
        "Test Accuracy": accuracy_score(y_test_np, preds_binary),
        "Test F1 Score": f1_score(y_test_np, preds_binary),
        "Test AUC": roc_auc_score(y_test_np, probs),
        "Test Precision": precision_score(y_test_np, preds_binary),
        "Test Recall": recall_score(y_test_np, preds_binary),
        "Confusion Matrix": confusion_matrix(y_test_np, preds_binary).tolist()
    }

    print("Validation of symbolic formula completed.")
    return metrics