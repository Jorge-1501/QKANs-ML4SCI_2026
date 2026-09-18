# Diagnostic tests for reports/baseline_quantum_collapse_investigation.md
"""
Read-only, non-training characterization tests that check the causes listed
in reports/baseline_quantum_collapse_investigation.md against artifacts
already produced by earlier pipeline runs (`outputs/`, `data/processed/`).

These are NOT unit tests of correct behavior -- the baseline QKAN currently
*is* biased toward one class, and these tests document/pin that fact so a
future fix can be checked against them (they should start failing once the
bias is actually corrected, at which point they should be updated or
removed). Nothing here calls QuantumKANTrainer.fit()/optimizer steps: the
only circuit execution is a single forward pass of an already warm-started
QKANModel over a small (<=32 row) slice of a cached preprocessed test set.
Everything is `pytest.skip`-ped if the seed/output artifacts it depends on
aren't present in the current checkout.
"""
import json
import re
from pathlib import Path

import pytest
import torch

from src.architectures.qkan_model import QKANModel
from src.utils import workspace

ROOT = Path(__file__).parent.parent.resolve()
FORWARD_BATCH_SIZE = 32


def _config(seed):
    """Paths come from workspace.get_config (the default mass-cut / n_subsets regime the
    investigation was run under) instead of being rebuilt by hand."""
    return workspace.get_config("top", seed)


def _canonical_subsets_path(seed):
    return Path(_config(seed)["canonical_cache_file"])


def _graph_path(seed):
    return Path(_config(seed)["polynomial_weights_dir"]) / "quantum_weights.pt"


def _chebyshev_report_path(seed):
    return Path(_config(seed)["Chebyshev_coefficients_path"])


def _baseline_metrics_paths():
    return sorted(
        path
        for run in workspace.iter_run_dirs("top")
        for path in run["path"].glob("results/qkan/*/baseline/metrics_qkan_baseline_*.json")
    )


def _history_loss_path(seed, backend="ideal"):
    return Path(_config(seed)[f"history_{backend}_loss"])


@pytest.mark.parametrize("seed", [3, 10])
def test_baseline_forward_pass_is_biased_toward_one_class(seed):
    """Cause #1/#2: a warm-started (untrained) QKANModel's raw PauliZ expval,
    even when tiny, gets pushed past the fixed 0.5 sigmoid threshold for
    nearly every sample because sigmoid([-1,1]) is compressed to
    [0.269, 0.731]. Forward-only, no gradients, batch capped at 32 rows."""
    graph_path = _graph_path(seed)
    subsets_path = _canonical_subsets_path(seed)
    if not graph_path.exists() or not subsets_path.exists():
        pytest.skip(f"cached artifacts for seed {seed} not present in this checkout")

    subsets = torch.load(subsets_path, weights_only=False)
    subset_idx = seed % subsets["n_subsets"]
    X_test = subsets["X_test_subsets"][subset_idx][:FORWARD_BATCH_SIZE]

    model = QKANModel(str(graph_path), backend_mode="ideal")
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        probs = torch.sigmoid(logits)

    assert torch.all(logits >= -1.0) and torch.all(logits <= 1.0), (
        "PauliZ expectation value must be bounded to [-1, 1]"
    )
    frac_positive = (probs > 0.5).float().mean().item()
    # Documents the current bug: almost every sample lands on the same side
    # of the threshold straight out of warm start, before any training step.
    assert frac_positive > 0.8 or frac_positive < 0.2, (
        f"seed {seed}: expected a strong positive/negative skew from warm "
        f"start (frac_positive={frac_positive:.3f}); if this now fails, the "
        "warm-start bias described in the report may have been fixed."
    )


@pytest.mark.parametrize("seed", [3, 10, 11, 12, 13])
def test_pruning_collapses_active_inputs_to_mass_and_multiplicity(seed):
    """Cause #3: classical pruning discards all 20 per-particle substructure
    features in every seed checked, leaving the quantum circuit only `m`
    (col 0) and `n` (col 1)."""
    report_path = _chebyshev_report_path(seed)
    if not report_path.exists():
        pytest.skip(f"chebyshev_coefficients.txt not present for seed {seed}")

    text = report_path.read_text()
    match = re.search(r"active_inputs \(raw\):\s*(\[[^\]]*\])", text)
    assert match, f"seed {seed}: 'active_inputs (raw): [...]' line not found in {report_path}"
    active_inputs = json.loads(match.group(1))

    assert active_inputs == [0, 1], (
        f"seed {seed}: expected active_inputs == [0, 1] (mass, multiplicity "
        f"only), got {active_inputs}; if this now fails, pruning may keep "
        "more features than at the time of the investigation."
    )


def test_baseline_metrics_show_systematic_positive_bias():
    """Cause #1: across every saved baseline metrics JSON, recall stays near
    1.0 and precision near 0.5 -- i.e. the model predicts "positive" for
    almost every sample regardless of the true label."""
    paths = _baseline_metrics_paths()
    if not paths:
        pytest.skip("no metrics_qkan_baseline_*.json files present in this checkout")

    for path in paths:
        metrics = json.loads(path.read_text())
        recall = metrics["Test Recall"]
        precision = metrics["Test Precision"]
        assert recall >= 0.9, (
            f"{path.relative_to(ROOT)}: expected recall >= 0.9 (documents "
            f"the current positive-class bias), got {recall:.3f}"
        )
        assert precision <= 0.6, (
            f"{path.relative_to(ROOT)}: expected precision <= 0.6 (documents "
            f"the current positive-class bias), got {precision:.3f}"
        )


def test_training_loss_stays_flat_near_random_baseline():
    """Cause #4: train/val loss for the trained (fine-tuned) QKAN barely
    moves away from ln(2) (~0.693, a coin flip's BCE loss) across the full
    training run, consistent with weak gradient signal."""
    history_path = _history_loss_path(seed=10, backend="ideal")
    if not history_path.exists():
        pytest.skip("history_loss.json not present for seed 10/ideal")

    history = json.loads(history_path.read_text())
    train_loss = history["train_loss"]
    assert train_loss, "train_loss history is empty"

    ln2 = 0.693147
    assert all(0.55 <= v <= 0.75 for v in train_loss), (
        f"expected every recorded train_loss value to stay within a narrow "
        f"band around ln(2)={ln2:.3f}; got min={min(train_loss):.3f}, "
        f"max={max(train_loss):.3f}"
    )
