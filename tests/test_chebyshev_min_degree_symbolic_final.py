# Approach B of the min-degree=3/max-degree=4 Chebyshev experiment (see
# reports/qkan_chebyshev_min_degree_experiment.md). Reconstructs the
# historical extraction path (commit 146e021: fitting against the SYMBOLIC
# branch of the fully fine-tuned 05_final checkpoint, i.e. "adjustment at the
# final classic part, after the final fine-tuning"), with the subnode_bias/
# subnode_scale shape fix already validated in
# reports/qkan_chebyshev_degree_comparison.md. Runs two configurations per
# seed: (1) the gated min-degree-3/max-degree-4 search used in Approach A
# (the literal request), and (2) the historical fixed-degree=4 fit with no
# R2 gate, to reproduce and cross-check report 2's own numbers for this
# branch. Does not modify src/architectures/extractor.py.
import os
import sys
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.chebyshev import chebfit, chebval
from sklearn.metrics import roc_auc_score

sys.path.append(str(Path(__file__).parent.parent.resolve()))

from src.utils import workspace
from src.architectures.extractor import SymbolicWarmStartExtractor
from src.architectures.qkan_model import QKANModel
import src.preprocessing.processor_top as processor

SEEDS = [12, 13, 14]
MIN_DEGREE = 3
MAX_DEGREE = 4

# seed 12's 05_final on disk predates the current hyperparams.py (a later
# --force rerun crashed with a NaN in symbolic fine-tuning under the current
# hyperparameters) -- flagged explicitly rather than silently reused.
STALE_FINAL_SEEDS = {12}


class MinDegreeSymbolicFinalExtractor(SymbolicWarmStartExtractor):
    """Fits against model.symbolic_fun (post fix_symbolic + fine-tuning)
    instead of model.act_fun, using subnode_bias/subnode_scale for the
    affine correction. _build_node_groups and extract_and_save's
    masking/grouping/padding are inherited unmodified: act_fun's mask still
    correctly reflects pruning at 05_final (pruning happens upstream of
    symbolic simplification/fine-tuning and is never undone), so the
    parent's act_fun-only mask check remains valid here.
    """

    def __init__(self, config, min_degree, max_degree):
        super().__init__(config)
        self.min_degree = min_degree
        self.max_degree = max_degree

    def _evaluate_isolated_edges(self, model, layer_index, input_index, output_index, x_vals):
        in_dim = int(model.width_in[layer_index])
        n = len(x_vals)

        x_zero = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var = torch.zeros((n, in_dim), dtype=torch.float32).to(self.device)
        x_var[:, input_index] = torch.tensor(x_vals, dtype=torch.float32).to(self.device)

        def layer_forward(x_in):
            try:
                symbolic = model.symbolic_fun[layer_index](x_in)
                x_out = symbolic[0] if isinstance(symbolic, tuple) else symbolic
            except Exception:
                out_dim = int(model.width_out[layer_index + 1])
                x_out = torch.zeros((n, out_dim), dtype=torch.float32).to(self.device)

            if hasattr(model, "subnode_bias") and model.subnode_bias is not None and len(model.subnode_bias) > layer_index:
                x_out = x_out + model.subnode_bias[layer_index]
            if hasattr(model, "subnode_scale") and model.subnode_scale is not None and len(model.subnode_scale) > layer_index:
                x_out = x_out * model.subnode_scale[layer_index]
            return x_out

        with torch.no_grad():
            y_var = layer_forward(x_var)[:, output_index].cpu().numpy()
            y_zero = layer_forward(x_zero)[:, output_index].cpu().numpy()

        y_var = np.nan_to_num(y_var, nan=0.0, posinf=0.0, neginf=0.0)
        y_zero = np.nan_to_num(y_zero, nan=0.0, posinf=0.0, neginf=0.0)
        return y_var - y_zero + (y_zero / in_dim)

    def _fit_edge(self, model, layer_index, input_index, output_index, x_vals):
        y_vals = self._evaluate_isolated_edges(model, layer_index, input_index, output_index, x_vals)
        dynamic_range = float(np.max(y_vals) - np.min(y_vals))
        ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2))
        ss_tot_safe = ss_tot if ss_tot > 1e-12 else 1e-12

        best_degree, best_coefs, best_r2 = None, None, -np.inf
        for degree in range(self.min_degree, self.max_degree + 1):
            coefs = chebfit(x_vals, y_vals, deg=degree)
            y_pred = chebval(x_vals, coefs)
            r2 = 1.0 - float(np.sum((y_vals - y_pred) ** 2)) / ss_tot_safe
            if r2 >= self.r2_threshold:
                return coefs.tolist(), dynamic_range, degree, r2
            if r2 > best_r2:
                best_degree, best_coefs, best_r2 = degree, coefs, r2

        return best_coefs.tolist(), dynamic_range, best_degree, best_r2


class FixedMaxDegreeSymbolicFinalExtractor(MinDegreeSymbolicFinalExtractor):
    """Same symbolic-branch/subnode-bias-fixed _evaluate_isolated_edges as
    above, but every edge is fit at exactly max_degree -- no R2 gate. This
    is the historical (commit 146e021) fixed-degree behavior, used here to
    reproduce reports/qkan_chebyshev_degree_comparison.md's
    'previous approach (subnode-bias fixed)' numbers as a cross-check."""

    def __init__(self, config, max_degree):
        SymbolicWarmStartExtractor.__init__(self, config)
        self.max_degree = max_degree

    def _fit_edge(self, model, layer_index, input_index, output_index, x_vals):
        y_vals = self._evaluate_isolated_edges(model, layer_index, input_index, output_index, x_vals)
        dynamic_range = float(np.max(y_vals) - np.min(y_vals))
        coefs = chebfit(x_vals, y_vals, deg=self.max_degree)
        y_pred = chebval(x_vals, coefs)
        ss_tot = float(np.sum((y_vals - np.mean(y_vals)) ** 2)) or 1e-12
        r2 = 1.0 - float(np.sum((y_vals - y_pred) ** 2)) / ss_tot
        return coefs.tolist(), dynamic_range, self.max_degree, r2


def degree_histogram(report_path):
    counts = {}
    with open(report_path) as f:
        for line in f:
            if "degree=" in line:
                deg = int(line.split("degree=")[1].split(",")[0])
                counts[str(deg)] = counts.get(str(deg), 0) + 1
    return counts


def evaluate_graph(graph, graph_path, seed):
    n_sum = sum(1 for h in graph["hidden_nodes"] if h["type"] == "sum")
    n_mult = sum(1 for h in graph["hidden_nodes"] if h["type"] == "mult")

    model = QKANModel(graph_path=graph_path, backend_mode="ideal")
    model.eval()

    _, _, _, _, X_test, y_test, _, _ = processor.load_and_preprocess_data(
        data_dir=os.path.join("data", "raw", "top"), task="top", seed=seed, force_process=False
    )

    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
    y_true = y_test.cpu().numpy().reshape(-1)

    auc = float(roc_auc_score(y_true, probs))
    flipped_auc = float(roc_auc_score(y_true, 1.0 - probs))
    identity_gap = abs(flipped_auc - (1.0 - auc))
    assert identity_gap < 1e-6, f"sign-flip identity broke (gap={identity_gap}) -- investigate"

    return {
        "n_qubits": graph["n_qubits"],
        "n_sum_nodes": n_sum,
        "n_mult_nodes": n_mult,
        "final_padded_degree": graph["degree"],
        "auc": auc,
        "flipped_auc": flipped_auc,
    }


def run_seed(seed, scratch_dir):
    print(f"\n{'=' * 60}\nSeed {seed} -- Approach B (symbolic / 05_final)\n{'=' * 60}")
    if seed in STALE_FINAL_SEEDS:
        print(f"[WARNING] seed {seed}'s 05_final checkpoint on disk predates the "
              f"current hyperparams.py (a fresh --force rerun previously crashed "
              f"with a NaN in symbolic fine-tuning). Using the existing checkpoint "
              f"as-is; treat this seed's Approach-B result with that caveat.")

    config = workspace.get_config(task="top", seed=seed)
    workspace.make_dirs(config)
    classic_model_path = os.path.join(config["final_model_path"], "05_final")

    # Configuration 1: gated min-degree-3/max-degree-4 search (as requested).
    gated_extractor = MinDegreeSymbolicFinalExtractor(config, min_degree=MIN_DEGREE, max_degree=MAX_DEGREE)
    gated_graph_path = os.path.join(scratch_dir, f"graph_seed{seed}_symbolicfinal_gated{MIN_DEGREE}-{MAX_DEGREE}.pt")
    gated_report_path = os.path.join(scratch_dir, f"report_seed{seed}_symbolicfinal_gated{MIN_DEGREE}-{MAX_DEGREE}.txt")
    gated_graph = gated_extractor.extract_and_save(classic_model_path, gated_graph_path, gated_report_path)
    gated_result = evaluate_graph(gated_graph, gated_graph_path, seed)
    gated_result["degree_histogram"] = degree_histogram(gated_report_path)

    # Configuration 2: historical fixed degree=4, no gate.
    fixed_extractor = FixedMaxDegreeSymbolicFinalExtractor(config, max_degree=MAX_DEGREE)
    fixed_graph_path = os.path.join(scratch_dir, f"graph_seed{seed}_symbolicfinal_fixed{MAX_DEGREE}.pt")
    fixed_report_path = os.path.join(scratch_dir, f"report_seed{seed}_symbolicfinal_fixed{MAX_DEGREE}.txt")
    fixed_graph = fixed_extractor.extract_and_save(classic_model_path, fixed_graph_path, fixed_report_path)
    fixed_result = evaluate_graph(fixed_graph, fixed_graph_path, seed)

    result = {
        "seed": seed,
        "approach": "B_symbolic_05_final",
        "min_degree": MIN_DEGREE,
        "max_degree": MAX_DEGREE,
        "stale_checkpoint": seed in STALE_FINAL_SEEDS,
        "gated_search": gated_result,
        "fixed_max_degree_no_gate": fixed_result,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="qkan_chebyshev_deg34_symbolicfinal_") as scratch_dir:
        results = [run_seed(seed, scratch_dir) for seed in SEEDS]

    print(f"\n{'=' * 60}\nSUMMARY -- Approach B (symbolic 05_final), degree {MIN_DEGREE}-{MAX_DEGREE}\n{'=' * 60}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
