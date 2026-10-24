"""Feature extractor: textbook values, invariances and diagnostic power."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pulse.features import FeatureExtractor
from pulse.signals.dataset import Dataset
from pulse.signals.generator import Condition, MachineSpec, WindowParams, generate_window


@pytest.fixture(scope="module")
def extractor(spec: MachineSpec) -> FeatureExtractor:
    return FeatureExtractor(spec)


def _window(condition: Condition, seed: int, spec: MachineSpec, rpm: float = 1800.0) -> np.ndarray:
    return generate_window(spec, WindowParams(condition, rpm, 0.9, 12.0), np.random.default_rng(seed)).signal


def test_feature_names_are_unique_and_match_vector_length(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    assert len(set(extractor.names)) == len(extractor.names) == extractor.n_features
    vec = extractor.extract(_window(Condition.HEALTHY, 0, spec), 1800.0)
    assert vec.shape == (extractor.n_features,)
    assert vec.dtype == np.float32


def test_all_features_finite_on_a_dataset(extractor: FeatureExtractor, small_test: Dataset) -> None:
    features = extractor.extract_batch(small_test.x, small_test.rpm)
    assert features.shape == (len(small_test), extractor.n_features)
    assert np.isfinite(features).all()


def test_kurtosis_of_gaussian_noise_is_about_three(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    x = np.random.default_rng(0).standard_normal(spec.n_samples)
    vec = extractor.extract(x, 1800.0)
    assert vec[extractor.names.index("kurtosis")] == pytest.approx(3.0, abs=0.4)


def test_crest_factor_of_a_sine_is_root_two(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    t = np.arange(spec.n_samples) / spec.fs_hz
    vec = extractor.extract(np.sin(2 * np.pi * 93.75 * t), 1800.0)
    assert vec[extractor.names.index("crest_factor")] == pytest.approx(np.sqrt(2.0), rel=0.01)


def test_band_energies_partition_the_spectrum(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    vec = extractor.extract(_window(Condition.MISALIGNMENT, 1, spec), 1800.0)
    bands = [10 ** vec[extractor.names.index(f"log_band_{i:02d}")] for i in range(16)]
    assert sum(bands) == pytest.approx(1.0, abs=0.01)


@settings(max_examples=25, deadline=None)
@given(st.floats(min_value=0.1, max_value=50.0))
def test_shape_features_are_scale_invariant_and_rms_is_linear(gain: float) -> None:
    spec = MachineSpec()
    extractor = FeatureExtractor(spec)
    x = _window(Condition.OUTER_RACE, 7, spec)
    base = extractor.extract(x, 1800.0)
    scaled = extractor.extract(x * gain, 1800.0)
    for name in ("crest_factor", "kurtosis", "shape_factor", "log_2x_over_1x", "env_bpfo_1", "spectral_entropy"):
        i = extractor.names.index(name)
        assert scaled[i] == pytest.approx(base[i], rel=1e-3, abs=1e-4)
    assert scaled[extractor.names.index("rms")] == pytest.approx(base[extractor.names.index("rms")] * gain, rel=1e-3)


@pytest.mark.parametrize(
    ("condition", "winner"),
    [(Condition.OUTER_RACE, "env_bpfo_1"), (Condition.INNER_RACE, "env_bpfi_1"), (Condition.BALL, "env_bsf_2")],
)
def test_envelope_features_point_at_the_right_fault(
    extractor: FeatureExtractor, spec: MachineSpec, condition: Condition, winner: str
) -> None:
    candidates = ["env_bpfo_1", "env_bpfi_1", "env_bsf_2"]
    hits = 0
    for seed in range(6):
        vec = extractor.extract(_window(condition, seed, spec), 1800.0)
        values = {name: vec[extractor.names.index(name)] for name in candidates}
        hits += max(values, key=values.get) == winner
    assert hits >= 5


def test_order_ratio_separates_imbalance_from_misalignment(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    i = extractor.names.index("log_2x_over_1x")
    imbalance = extractor.extract(_window(Condition.IMBALANCE, 2, spec), 1800.0)[i]
    misalignment = extractor.extract(_window(Condition.MISALIGNMENT, 2, spec), 1800.0)[i]
    assert imbalance < 0.0 < misalignment


def test_extract_is_deterministic(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    x = _window(Condition.BALL, 4, spec)
    np.testing.assert_array_equal(extractor.extract(x, 1800.0), extractor.extract(x, 1800.0))


def test_extract_validates_inputs(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    with pytest.raises(ValueError):
        extractor.extract(np.zeros(spec.n_samples + 1), 1800.0)
    with pytest.raises(ValueError):
        extractor.extract(np.zeros(spec.n_samples), 0.0)


def test_constant_window_does_not_produce_nan(extractor: FeatureExtractor, spec: MachineSpec) -> None:
    assert np.isfinite(extractor.extract(np.zeros(spec.n_samples), 1800.0)).all()
