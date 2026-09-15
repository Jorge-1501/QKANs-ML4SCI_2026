# Approach A of the min-degree=3/max-degree=4 Chebyshev experiment (see
# reports/qkan_chebyshev_min_degree_experiment.md). On the CURRENT extraction
# path (numeric b-spline branch of the 03_retrained checkpoint), for seeds
# 12/13/14, this runs two configurations:
#   1. GATED search: degree in [3, 4], accept the first that clears the R2
#      threshold (the literal "min degree 3, max degree 4" request).
#   2. FIXED max-degree, no gate: every edge is always fit at degree 4,
#      never early-accepted at 3 -- added because (1) turned out to make no
#      difference (every edge already clears the R2 gate right at degree 3,
#      so degree 4 is never tried), while this configuration is what
#      actually recovers baseline AUC (see the report for the numbers).
# Does not modify src/architectures/extractor.py -- both configurations are
# implemented by subclassing and overriding only _fit_edge.
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


class MinDegreeExtractor(SymbolicWarmStartExtractor):
    """SymbolicWarmStartExtractor with a configurable minimum-degree floor.

    Only _fit_edge is overridden -- _evaluate_isolated_edges,
    _build_node_groups and extract_and_save are inherited unmodified, since
    they already read act_fun/subnode_bias/subnode_scale correctly for the
    03_retrained checkpoint and are degree-agnostic.
    """

    def __init__(self, config, min_degree, max_degree):
        super().__init__(config)
        self.min_degree = min_degree
        self.max_degree = max_degree

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


class FixedMaxDegreeExtractor(SymbolicWarmStartExtractor):
    """Every edge is fit at exactly self.max_degree -- no R2 gate, no early
    acceptance at a lower degree. This is NOT what the extractor's own
    docstring describes (it documents the gated minimum-degree search), but
    it mirrors the historical (pre-search) fixed-degree behavior."""

    def __init__(self, config, max_degree):
        super().__init__(config)
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
    """Counts 'degree=N' occurrences in the extractor's own text report --
    these are the PRE-padding, per-edge chosen degrees (padding only touches
    the coefficient list, never the recorded 'degree' field)."""
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
    print(f"\n{'=' * 60}\nSeed {seed} -- Approach A (b-spline / 03_retrained)\n{'=' * 60}")

    config = workspace.get_config(task="top", seed=seed)
    workspace.make_dirs(config)
    classic_model_path = os.path.join(config["retrained_model_path"], "03_retrained")

    # Configuration 1: gated min-degree-3/max-degree-4 search (as requested).
    gated_extractor = MinDegreeExtractor(config, min_degree=MIN_DEGREE, max_degree=MAX_DEGREE)
    gated_graph_path = os.path.join(scratch_dir, f"graph_seed{seed}_bspline_gated{MIN_DEGREE}-{MAX_DEGREE}.pt")
    gated_report_path = os.path.join(scratch_dir, f"report_seed{seed}_bspline_gated{MIN_DEGREE}-{MAX_DEGREE}.txt")
    gated_graph = gated_extractor.extract_and_save(classic_model_path, gated_graph_path, gated_report_path)
    gated_result = evaluate_graph(gated_graph, gated_graph_path, seed)
    gated_result["degree_histogram"] = degree_histogram(gated_report_path)

    # Configuration 2: fixed degree=4, no gate -- added after (1) showed no
    # effect (every edge already clears R2>=0.95 right at degree 3).
    fixed_extractor = FixedMaxDegreeExtractor(config, max_degree=MAX_DEGREE)
    fixed_graph_path = os.path.join(scratch_dir, f"graph_seed{seed}_bspline_fixed{MAX_DEGREE}.pt")
    fixed_report_path = os.path.join(scratch_dir, f"report_seed{seed}_bspline_fixed{MAX_DEGREE}.txt")
    fixed_graph = fixed_extractor.extract_and_save(classic_model_path, fixed_graph_path, fixed_report_path)
    fixed_result = evaluate_graph(fixed_graph, fixed_graph_path, seed)

    result = {
        "seed": seed,
        "approach": "A_bspline_03_retrained",
        "min_degree": MIN_DEGREE,
        "max_degree": MAX_DEGREE,
        "gated_search": gated_result,
        "fixed_max_degree_no_gate": fixed_result,
    }
    print(json.dumps(result, indent=2))
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="qkan_chebyshev_deg34_bspline_") as scratch_dir:
        results = [run_seed(seed, scratch_dir) for seed in SEEDS]

    print(f"\n{'=' * 60}\nSUMMARY -- Approach A (b-spline), degree {MIN_DEGREE}-{MAX_DEGREE}\n{'=' * 60}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
