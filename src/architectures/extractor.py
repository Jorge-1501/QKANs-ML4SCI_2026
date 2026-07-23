# src/architectures/extractor.py
import os
import torch
import numpy as np
from numpy.polynomial.chebyshev import chebfit, cheb2poly
from kan import KAN
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.architectures.hep_kan import HEPKAN

class SymbolicWarmStartExtractor:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.degree = self.config.get("chebyshev_degree", 4)
        
    def _evaluate_isolated_edges(self, model, layer_index, input_index, output_index, x_vals):
        """
        Isolates the effect of a single input variable on a specific output node.
        """
        layer_width = model.width[layer_index]
        in_dim = layer_width[0] if isinstance(layer_width, list) else layer_width
        n = len(x_vals)

        x_zero = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var[:, input_index] = torch.tensor(x_vals, dtype=torch.float32).to(self.device)

        def layer_forward(x_in):
            try:
                symbolic = model.symbolic_fun[layer_index](x_in)
                x_out = symbolic[0] if isinstance(symbolic, tuple) else symbolic
            except Exception:
                x_out = torch.zeros((n, model.width[layer_index + 1][0]), dtype=torch.float32).to(self.device)

            if hasattr(model, "node_bias") and model.node_bias is not None and len(model.node_bias) > layer_index:
                x_out += model.node_bias[layer_index]
            if hasattr(model, "node_scale") and model.node_scale is not None and len(model.node_scale) > layer_index:
                x_out *= model.node_scale[layer_index]
            return x_out
        
        with torch.no_grad():
            y_var = layer_forward(x_var)[:, output_index].cpu().numpy()
            y_zero = layer_forward(x_zero)[:, output_index].cpu().numpy()

        y_var = np.nan_to_num(y_var, nan=0.0, posinf=0.0, neginf=0.0)
        y_zero = np.nan_to_num(y_zero, nan=0.0, posinf=0.0, neginf=0.0)
        return y_var - y_zero + (y_zero / in_dim)

    def extract_and_save(self, classic_model_path, output_weights_path, report_path):
        """
        Extracts the functions from the classical KAN, filters inactive inputs
        by evaluating their actual dynamic range, and generates the quantum weights file.
        """
        print("\n" + "="*40)
        print("[Extractor] Starting Dynamic Extraction and Filtering by Dynamic Range")
        print("="*40)

        # 1. Load the trained classical model
        base_kan = HEPKAN.loadckpt(classic_model_path)
        model = HEPKAN.__new__(HEPKAN)
        model.__dict__.update(base_kan.__dict__)
        model.to(self.device)
        model.eval()

        in_dim = model.width_in[0]
        active_inputs = []
        x_vals = np.linspace(-1, 1, 500)
        variance_threshold = 1e-3  # Umbral mínimo de variación ($\Delta y$) para considerar una variable activa

        print("[Extractor] Evaluating the physical relevance of each input variable...")
        for i in range(in_dim):
            # Comprobación de máscara dura de PyTorch
            mask_act = model.act_fun[0].mask[i, :].sum().item()
            mask_sym = model.symbolic_fun[0].mask[:, i].sum().item() if hasattr(model, "symbolic_fun") else 0.0

            if mask_act == 0.0 and mask_sym == 0.0:
                print(f" -> Variable {i}: OMITTED (Mask at absolute 0.0)")
                continue

            # Check by Dynamic Range ($\Delta y = \max(y) - \min(y)$)
            y_eval = self._evaluate_isolated_edges(model, layer_index=0, input_index=i, output_index=0, x_vals=x_vals)
            dynamic_range = float(np.max(y_eval) - np.min(y_eval))

            if dynamic_range > variance_threshold:
                active_inputs.append(i)
                print(f" -> Variable {i} ACTIVE (Dynamic range: {dynamic_range:.5f})")
            else:
                print(f" -> Variable {i} DISCARDED (Insignificant contribution: {dynamic_range:.6f})")

        # Safety fallback in case extreme pruning discards everything
        if not active_inputs:
            print("[Extractor] Warning: No variable surpassed the threshold. Selecting the one with the highest impact...")
            ranges = [
                float(np.max(self._evaluate_isolated_edges(model, 0, i, 0, x_vals)) - 
                      np.min(self._evaluate_isolated_edges(model, 0, i, 0, x_vals))) 
                for i in range(in_dim)
            ]
            active_inputs = [int(np.argmax(ranges))]

        n_qubits = len(active_inputs)
        print(f"\n[Extractor] Filtering completed: {n_qubits} Qubits assigned (Inputs: {active_inputs})")

        # Extract polynomial coefficients for active qubits
        quantum_weights = np.zeros((n_qubits, self.degree + 1))
        for q_idx, classical_in_idx in enumerate(active_inputs):
            y_vals = self._evaluate_isolated_edges(model, layer_index=0, input_index=classical_in_idx, output_index=0, x_vals=x_vals)
            coefs = chebfit(x_vals, y_vals, deg=self.degree)
            quantum_weights[q_idx, :] = coefs

        # Extract output layer scale
        y_out = self._evaluate_isolated_edges(model, layer_index=1, input_index=0, output_index=0, x_vals=x_vals)
        out_coefs = chebfit(x_vals, y_out, deg=1)

        # Export using PyTorch Tensors to avoid incompatibilities with PyTorch 2.6+
        os.makedirs(os.path.dirname(output_weights_path), exist_ok=True)
        export_data = {
            "active_inputs": active_inputs,
            "n_qubits": n_qubits,
            "degree": self.degree,
            "edge_weights": torch.tensor(quantum_weights, dtype=torch.float32),
            "out_weights": torch.tensor(out_coefs, dtype=torch.float32)
        }
        torch.save(export_data, output_weights_path)
        print(f"[Extractor] Weights and metadata successfully exported to: {output_weights_path}")

        # Generate written report
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w") as f:
            f.write("=== DYNAMIC WEIGHTS EXTRACTION REPORT ===\n")
            f.write(f"Required Qubits (n_qubits): {n_qubits}\n")
            f.write(f"Selected Active Variables: {active_inputs}\n")
            f.write(f"Polynomial Degree (Chebyshev): {self.degree}\n\n")
            for q_idx, c_idx in enumerate(active_inputs):
                f.write(f"Wire {q_idx} (Classical Input {c_idx}): {quantum_weights[q_idx].tolist()}\n")

        return export_data