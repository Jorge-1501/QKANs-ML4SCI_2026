# eval_sine_baseline.py
"""
Untrained (pre-fine-tuning), ideal-device evaluation of three QKAN initializations
on the SAME pruned graph and SAME test set, per seed:

  1. sine      : SineKAN-basis warm start (SineWarmStartExtractor)
  2. chebyshev : Chebyshev warm start (SymbolicWarmStartExtractor), re-extracted so
                 all three run through the identical pipeline
  3. random    : Chebyshev graph structure, weights ~ N(0, 1), no training

No training happens here. Outputs land in the run's outputs dir
(outputs/top/<cut>/<run>/seed_<N>/results/qkan/ideal/{sine_baseline,baseline,baseline_random}/).

--classic_root points at where each seed's 03_retrained checkpoint lives:
    <classic_root>/seed_<N>/models/03_retrained/03_retrained
(default outputs/top: the pre-refactor flat layout that seeds 10-14 still use).
"""
import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from src.utils import workspace
from src.architectures.extractor import SymbolicWarmStartExtractor
from src.architectures.extractor_sine import SineWarmStartExtractor
from src.architectures.quantum_kan import QuantumKANTrainer
import src.preprocessing.processor_top as processor_top


def run_seed(seed, task, classic_root):
    CONFIG = workspace.get_config(task=task, seed=seed)
    workspace.make_dirs(CONFIG)

    _, _, _, _, X_test, y_test, _, _ = processor_top.load_and_preprocess_data(
        data_dir=os.path.join(CONFIG["raw_data_dir"], task),
        task=task, force_process=False, seed=seed, full_dataset=False,
    )
    print(f"[seed {seed}] test set: {len(X_test)} samples, subset {CONFIG['subset_id']}")

    classic_ckpt = os.path.join(classic_root, f"seed_{seed}", "models", "03_retrained", "03_retrained")
    quantum_dir = CONFIG["polynomial_weights_dir"]
    cheb_graph = os.path.join(quantum_dir, "quantum_weights.pt")
    sine_graph = os.path.join(quantum_dir, CONFIG["quantum_graph_sine_filename"])
    os.makedirs(quantum_dir, exist_ok=True)

    SymbolicWarmStartExtractor(CONFIG).extract_and_save(
        classic_ckpt, cheb_graph, os.path.join(CONFIG["results_dir"], "chebyshev_coefficients.txt"))
    SineWarmStartExtractor(CONFIG).extract_and_save(
        classic_ckpt, sine_graph, os.path.join(CONFIG["results_dir"], "sine_coefficients.txt"))

    out = {}

    trainer = QuantumKANTrainer(CONFIG, train_backend="ideal", graph_filename=CONFIG["quantum_graph_sine_filename"])
    out["sine"] = trainer.evaluate_baseline(X_test, y_test, eval_backend="ideal", sine=True)

    trainer = QuantumKANTrainer(CONFIG, train_backend="ideal")
    out["chebyshev"] = trainer.evaluate_baseline(X_test, y_test, eval_backend="ideal")

    workspace.set_seed(seed, purpose="untrained random VQC init")
    trainer = QuantumKANTrainer(CONFIG, train_backend="ideal", random_init=True)
    out["random"] = trainer.evaluate_baseline(X_test, y_test, eval_backend="ideal", random_init=True)
    return out


def main(args):
    all_out = {}
    for seed in args.seeds:
        all_out[seed] = run_seed(seed, args.task, args.classic_root)

    summary = {s: {k: {m: v[m] for m in ("Test AUC", "Test Accuracy", "Test F1 Score",
                                          "Test Precision", "Test Recall", "Confusion Matrix")}
                   for k, v in d.items()} for s, d in all_out.items()}
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {args.summary}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Untrained ideal-device QKAN baseline: sine vs chebyshev vs random.")
    p.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    p.add_argument("--task", default="top")
    p.add_argument("--classic_root", default=os.path.join(workspace.get_project_root(), "outputs", "top"))
    p.add_argument("--summary", default=os.path.join(workspace.get_project_root(), "outputs", "top", "aggregate",
                                                     "sine_vs_chebyshev_vs_random_baseline.json"))
    args = p.parse_args()
    try:
        main(args)
    except Exception as e:
        print(f"A fatal error occurred: {e}")
        traceback.print_exc()
        sys.exit(1)
