"""Baseline and CNN: learning, determinism, persistence, interface contracts."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from pulse.models.baseline import BaselineConfig, FeatureBaseline
from pulse.models.cnn import PulseCNN, Standardize, TrainConfig, load_cnn, predict_logits, save_cnn, train_cnn
from pulse.signals.dataset import Dataset, DatasetConfig, make_dataset
from pulse.signals.generator import CLASS_NAMES, MachineSpec


@pytest.fixture(scope="module")
def baseline(spec: MachineSpec, small_train: Dataset) -> FeatureBaseline:
    return FeatureBaseline(spec, BaselineConfig(max_iter=60)).fit(small_train)


def test_baseline_learns_well_above_chance(baseline: FeatureBaseline, small_test: Dataset) -> None:
    accuracy = float((baseline.predict(small_test.x, small_test.rpm) == small_test.y).mean())
    assert accuracy > 0.7


def test_baseline_probabilities_are_distributions(baseline: FeatureBaseline, small_test: Dataset) -> None:
    probs = baseline.predict_proba(small_test.x[:10], small_test.rpm[:10])
    assert probs.shape == (10, len(CLASS_NAMES))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_baseline_save_load_roundtrip(baseline: FeatureBaseline, small_test: Dataset, tmp_path: Path) -> None:
    size = baseline.save(tmp_path / "b.pkl")
    assert size > 1000
    loaded = FeatureBaseline.load(tmp_path / "b.pkl")
    np.testing.assert_array_equal(
        loaded.predict_proba(small_test.x[:5], small_test.rpm[:5]),
        baseline.predict_proba(small_test.x[:5], small_test.rpm[:5]),
    )


def test_baseline_load_rejects_foreign_pickles(tmp_path: Path) -> None:
    (tmp_path / "junk.pkl").write_bytes(pickle.dumps({"not": "a model"}))
    with pytest.raises(TypeError):
        FeatureBaseline.load(tmp_path / "junk.pkl")


def test_cnn_output_shape_and_parameter_budget(spec: MachineSpec) -> None:
    model = PulseCNN().eval()
    out = model(torch.zeros(3, spec.n_samples))
    assert out.shape == (3, len(CLASS_NAMES))
    assert model.n_parameters() < 100_000


def test_cnn_accepts_2d_and_3d_input_identically(spec: MachineSpec) -> None:
    torch.manual_seed(0)
    model = PulseCNN().eval()
    x = torch.randn(2, spec.n_samples)
    with torch.no_grad():
        torch.testing.assert_close(model(x), model(x.unsqueeze(1)))


def test_standardize_removes_scale_and_reports_it() -> None:
    x = torch.randn(4, 1, 512) * 3.0 + 7.0
    z, log_scale = Standardize()(x)
    torch.testing.assert_close(z.mean(dim=-1), torch.zeros(4, 1), atol=1e-4, rtol=0)
    torch.testing.assert_close(z.std(dim=-1, unbiased=False), torch.ones(4, 1), atol=1e-3, rtol=0)
    assert log_scale.shape == (4, 1)
    assert torch.allclose(log_scale.exp(), x.std(dim=-1, unbiased=False), rtol=1e-3)


def test_training_is_deterministic_for_a_fixed_seed() -> None:
    data = make_dataset(DatasetConfig(n_per_class=6, seed=5))
    config = TrainConfig(epochs=2, threads=2, seed=3, val_fraction=0.0, batch_size=16)
    model_a, history_a = train_cnn(data, config)
    model_b, history_b = train_cnn(data, config)
    for (name, a), (_, b) in zip(model_a.state_dict().items(), model_b.state_dict().items(), strict=True):
        assert torch.equal(a, b), name
    assert history_a.train_loss == history_b.train_loss


def test_training_reduces_loss_and_keeps_best_epoch(small_train: Dataset) -> None:
    model, history = train_cnn(small_train, TrainConfig(epochs=3, threads=2, seed=1, val_fraction=0.2))
    assert len(history.train_loss) == 3
    assert history.train_loss[-1] < history.train_loss[0]
    assert 0 <= history.best_epoch < 3
    assert max(history.val_accuracy) == history.val_accuracy[history.best_epoch]
    assert isinstance(model, PulseCNN)


def test_cnn_save_load_roundtrip(tiny_model: PulseCNN, small_test: Dataset, tmp_path: Path) -> None:
    size = save_cnn(tiny_model, tmp_path / "m.pt")
    assert size > 100_000
    loaded = load_cnn(tmp_path / "m.pt")
    np.testing.assert_allclose(
        predict_logits(loaded, small_test.x[:8]), predict_logits(tiny_model, small_test.x[:8]), atol=1e-6
    )


def test_predict_logits_is_batch_size_independent(tiny_model: PulseCNN, small_test: Dataset) -> None:
    whole = predict_logits(tiny_model, small_test.x[:20], batch_size=20)
    pieces = predict_logits(tiny_model, small_test.x[:20], batch_size=7)
    np.testing.assert_allclose(whole, pieces, atol=1e-5)
