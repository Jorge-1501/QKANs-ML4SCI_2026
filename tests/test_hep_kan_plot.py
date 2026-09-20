# Covers HEPKAN.plot input labelling when len(in_vars) != width_in[0].
"""
Regression tests for the IndexError in HEPKAN.plot() when the model has more
inputs than names in `in_vars` (e.g. width[0] > 22 while the feature-name list
was hardcoded to 22), plus correct labelling of pruned models.
"""
import types

import pytest
import torch

from src.architectures.hep_kan import HEPKAN
from src.utils.hyperparams import build_feature_names, get_hyperparams


def _model_with_data(n_in, hidden=(2, 0)):
    model = HEPKAN(width=[n_in, list(hidden), 1], grid=3, k=3, seed=0, auto_save=False)
    x = torch.rand(64, n_in) * 2 - 1
    model(x)
    return model


def test_plot_with_more_inputs_than_names(tmp_path):
    model = _model_with_data(30)
    model.plot(
        folder=str(tmp_path / "edges"),
        save_path=str(tmp_path / "graph.png"),
        in_vars=build_feature_names(22),  # 22 names for 30 inputs
        varscale=0.5,
    )
    assert (tmp_path / "graph.png").exists()


def test_plot_with_matching_names(tmp_path):
    model = _model_with_data(30)
    model.plot(
        folder=str(tmp_path / "edges"),
        save_path=str(tmp_path / "graph.png"),
        in_vars=build_feature_names(30),
        varscale=0.5,
    )
    assert (tmp_path / "graph.png").exists()


def test_resolve_in_vars_pads_missing_names():
    model = types.SimpleNamespace()
    labels = HEPKAN._resolve_in_vars(model, ["a", "b"], 4)
    assert labels == ["a", "b", "x_3", "x_4"]


def test_resolve_in_vars_truncates_extra_names():
    model = types.SimpleNamespace()
    assert HEPKAN._resolve_in_vars(model, ["a", "b", "c"], 2) == ["a", "b"]


def test_resolve_in_vars_uses_surviving_ids_for_pruned_model():
    names = build_feature_names(22)
    model = types.SimpleNamespace(input_id=torch.tensor([0, 5, 9]))
    assert HEPKAN._resolve_in_vars(model, names, 3) == [names[0], names[5], names[9]]


def test_resolve_in_vars_ignores_stale_input_id():
    # input_id points past the name list -> fall back to padding, no crash
    model = types.SimpleNamespace(input_id=torch.tensor([0, 40]))
    assert HEPKAN._resolve_in_vars(model, ["a", "b"], 3) == ["a", "b", "x_3"]


@pytest.mark.parametrize("n_inputs", [2, 22, 30, 42])
def test_build_feature_names_length(n_inputs):
    names = build_feature_names(n_inputs)
    assert len(names) == n_inputs
    assert names[:2] == ["m", "n"]


@pytest.mark.parametrize("bad", [1, 0, 23])
def test_build_feature_names_rejects_invalid(bad):
    with pytest.raises(ValueError):
        build_feature_names(bad)


def test_config_width_matches_feature_names():
    cfg = get_hyperparams()
    assert cfg["width"][0] == len(cfg["features"])
