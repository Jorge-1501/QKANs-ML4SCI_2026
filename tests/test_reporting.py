# Covers src/utils/reporting.py: metrics collection into a long-format table.
# _flatten_metrics_file is tested directly against a real temp JSON file;
# compute_run_statistics is tested against a small synthetic outputs/<task>/seed_*/
# tree built via workspace.get_config's own path keys, so this stays valid even if
# the underlying directory layout changes. No real pipeline run required.
import json
import os

from src.utils import reporting, workspace


def test_flatten_metrics_file_missing_path_returns_none(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    assert reporting._flatten_metrics_file(str(missing), tags={"model": "x"}) is None


def test_flatten_metrics_file_merges_tags_and_serializes_nested_values(tmp_path):
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({
        "Test AUC": 0.91,
        "Test Accuracy": 0.88,
        "Confusion Matrix": [[10, 2], [1, 9]],
        "Backend": "ideal",
    }))

    row = reporting._flatten_metrics_file(
        str(metrics_path), tags={"task": "top", "seed": 3, "subset_id": 3, "model": "qkan_ideal"}
    )

    assert row["task"] == "top"
    assert row["seed"] == 3
    assert row["subset_id"] == 3
    assert row["model"] == "qkan_ideal"
    assert row["Test AUC"] == 0.91
    assert row["Backend"] == "ideal"
    # Nested list value must be JSON-stringified, not left as a Python list.
    assert row["Confusion Matrix"] == json.dumps([[10, 2], [1, 9]])


def test_compute_run_statistics_collects_only_existing_files_tagged_by_model(tmp_path, monkeypatch):
    monkeypatch.setattr(workspace, "get_project_root", lambda: tmp_path)

    task = "top"
    for seed in (0, 1):
        config = workspace.get_config(task, seed)

        base_path = config["base_eval_metrics"]
        os.makedirs(os.path.dirname(base_path), exist_ok=True)
        with open(base_path, "w") as f:
            json.dump({"Test AUC": 0.5 + seed * 0.01}, f)

        qkan_path = config["metrics_qkan_ideal"]
        os.makedirs(os.path.dirname(qkan_path), exist_ok=True)
        with open(qkan_path, "w") as f:
            json.dump({"Test AUC": 0.6 + seed * 0.01, "Backend": "ideal"}, f)

        # Every other registry entry (retrained/symbolic/final/noisy/shots/baseline...)
        # is deliberately left missing to exercise the partial-sweep skip path.

    df = reporting.compute_run_statistics(task)

    # 2 seeds x 2 present stages = 4 rows, nothing else.
    assert len(df) == 4
    assert set(df["model"].unique()) == {"classical_base", "qkan_ideal"}
    assert set(df["seed"].unique()) == {0, 1}

    # subset_id = seed % n_subsets (default n_subsets=15) for every row.
    for _, row in df.iterrows():
        assert row["subset_id"] == row["seed"] % 15

    # Pure collection: no mean/std/aggregate columns anywhere in the output.
    assert not any(col.lower() in ("mean", "std") for col in df.columns)

    base_rows = df[df["model"] == "classical_base"].sort_values("seed")
    assert list(base_rows["Test AUC"]) == [0.5, 0.51]
