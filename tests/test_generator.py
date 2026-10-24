"""The signal generator must obey the physics it claims to model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import signal as sps

from pulse.signals.bearing import fault_frequencies
from pulse.signals.dataset import Dataset, DatasetConfig, make_dataset
from pulse.signals.generator import CLASS_NAMES, Condition, MachineSpec, WindowParams, generate_window


def _clean(condition: Condition, rpm: float = 1800.0, seed: int = 0, **kwargs: float) -> tuple[np.ndarray, MachineSpec]:
    spec = MachineSpec()
    params = WindowParams(condition, rpm, kwargs.pop("severity", 1.0), kwargs.pop("snr_db", 30.0), **kwargs)
    return generate_window(spec, params, np.random.default_rng(seed)), spec


def _spectrum_peak_hz(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    mag = np.abs(np.fft.rfft((x - x.mean()) * np.hanning(x.size)))
    mask = (freqs >= lo) & (freqs <= hi)
    return float(freqs[mask][np.argmax(mag[mask])])


def _envelope_peak_hz(x: np.ndarray, spec: MachineSpec) -> float:
    sos = sps.butter(
        4, [spec.resonance_hz * 0.6, spec.resonance_hz * 1.4], btype="bandpass", fs=spec.fs_hz, output="sos"
    )
    env = np.abs(sps.hilbert(sps.sosfiltfilt(sos, x - x.mean())))
    return _spectrum_peak_hz(env, spec.fs_hz, 40.0, 1000.0)


def test_window_shape_dtype_and_full_scale() -> None:
    w, spec = _clean(Condition.HEALTHY)
    assert w.signal.shape == (spec.n_samples,)
    assert w.signal.dtype == np.float32
    assert np.abs(w.signal).max() <= spec.full_scale_g


def test_same_seed_gives_identical_window() -> None:
    a, _ = _clean(Condition.INNER_RACE, seed=3)
    b, _ = _clean(Condition.INNER_RACE, seed=3)
    np.testing.assert_array_equal(a.signal, b.signal)


def test_different_seed_gives_different_window() -> None:
    a, _ = _clean(Condition.INNER_RACE, seed=3)
    b, _ = _clean(Condition.INNER_RACE, seed=4)
    assert not np.array_equal(a.signal, b.signal)


def test_outer_race_envelope_spectrum_peaks_at_bpfo() -> None:
    w, spec = _clean(Condition.OUTER_RACE, speed_drift=0.0, slip=0.0)
    expected = fault_frequencies(spec.bearing, 30.0).bpfo
    assert _envelope_peak_hz(w.signal, spec) == pytest.approx(expected, rel=0.06)


def test_inner_race_impacts_are_spaced_at_bpfi() -> None:
    w, spec = _clean(Condition.INNER_RACE, speed_drift=0.0, slip=0.0)
    spacing = np.median(np.diff(w.impact_times_s))
    assert spacing == pytest.approx(1.0 / fault_frequencies(spec.bearing, 30.0).bpfi, rel=0.02)


def test_ball_fault_impacts_are_spaced_at_twice_bsf() -> None:
    w, spec = _clean(Condition.BALL, speed_drift=0.0, slip=0.0)
    spacing = np.median(np.diff(w.impact_times_s))
    assert spacing == pytest.approx(1.0 / (2.0 * fault_frequencies(spec.bearing, 30.0).bsf), rel=0.02)


def test_slip_jitters_impact_spacing() -> None:
    steady, _ = _clean(Condition.OUTER_RACE, speed_drift=0.0, slip=0.0)
    jittery, _ = _clean(Condition.OUTER_RACE, speed_drift=0.0, slip=0.05)
    assert np.std(np.diff(jittery.impact_times_s)) > 5 * np.std(np.diff(steady.impact_times_s))


def test_imbalance_spectrum_is_dominated_by_shaft_frequency() -> None:
    w, spec = _clean(Condition.IMBALANCE, speed_drift=0.0)
    assert _spectrum_peak_hz(w.signal, spec.fs_hz, 5.0, 500.0) == pytest.approx(30.0, abs=12.0)


def test_misalignment_spectrum_is_dominated_by_twice_shaft_frequency() -> None:
    w, spec = _clean(Condition.MISALIGNMENT, speed_drift=0.0)
    assert _spectrum_peak_hz(w.signal, spec.fs_hz, 5.0, 500.0) == pytest.approx(60.0, abs=12.0)


def test_imbalance_amplitude_scales_with_speed_squared() -> None:
    def one_x_amplitude(rpm: float) -> float:
        w, spec = _clean(Condition.IMBALANCE, rpm=rpm, speed_drift=0.0, seed=11)
        freqs = np.fft.rfftfreq(w.signal.size, 1.0 / spec.fs_hz)
        mag = np.abs(np.fft.rfft(w.signal - w.signal.mean()))
        band = (freqs > rpm / 60 * 0.9) & (freqs < rpm / 60 * 1.1)
        return float(mag[band].max())

    ratio = one_x_amplitude(3600.0) / one_x_amplitude(1800.0)
    assert 3.0 < ratio < 5.0


def test_lower_snr_means_more_noise_power() -> None:
    quiet, _ = _clean(Condition.HEALTHY, snr_db=20.0, seed=5)
    noisy, _ = _clean(Condition.HEALTHY, snr_db=-6.0, seed=5)
    assert noisy.signal.std() > 2.0 * quiet.signal.std()


def test_speed_profile_stays_within_drift_bound() -> None:
    w, _ = _clean(Condition.HEALTHY, speed_drift=0.03)
    assert np.all(w.shaft_hz >= 30.0 * 0.97 - 1e-9)
    assert np.all(w.shaft_hz <= 30.0 * 1.03 + 1e-9)
    assert w.shaft_hz.std() > 0.0


def test_bearing_faults_are_impulsive_and_healthy_is_near_gaussian() -> None:
    def kurtosis(x: np.ndarray) -> float:
        c = x - x.mean()
        return float(np.mean(c**4) / np.mean(c**2) ** 2)

    healthy = [kurtosis(_clean(Condition.HEALTHY, seed=s, snr_db=10.0)[0].signal) for s in range(5)]
    faulty = [kurtosis(_clean(Condition.OUTER_RACE, seed=s, snr_db=10.0)[0].signal) for s in range(5)]
    assert max(healthy) < 3.5
    assert min(faulty) > 4.0


def test_adc_quantises_to_its_step() -> None:
    w, spec = _clean(Condition.HEALTHY)
    step = 2.0 * spec.full_scale_g / 2**spec.adc_bits
    ratio = w.signal.astype(np.float64) / step
    assert np.allclose(ratio, np.round(ratio), atol=1e-3)


@pytest.mark.parametrize("kwargs", [{"rpm": 0.0}, {"severity": 1.5}, {"speed_drift": 0.6}])
def test_invalid_window_params_are_rejected(kwargs: dict[str, float]) -> None:
    base = {"condition": Condition.HEALTHY, "rpm": 1800.0, "severity": 0.5, "snr_db": 5.0}
    with pytest.raises(ValueError):
        WindowParams(**{**base, **kwargs})


def test_invalid_machine_spec_is_rejected() -> None:
    with pytest.raises(ValueError):
        MachineSpec(resonance_hz=7000.0)
    with pytest.raises(ValueError):
        MachineSpec(damping_ratio=1.5)


def test_dataset_is_balanced_shuffled_and_in_range() -> None:
    ds = make_dataset(DatasetConfig(n_per_class=12, seed=9))
    assert len(ds) == 12 * len(CLASS_NAMES)
    assert np.bincount(ds.y).tolist() == [12] * len(CLASS_NAMES)
    assert ds.y[:12].tolist() != [0] * 12
    assert ds.rpm.min() >= 900.0 and ds.rpm.max() <= 3600.0
    assert ds.severity.min() >= 0.05 and ds.snr_db.max() <= 15.0


def test_dataset_generation_is_deterministic() -> None:
    a = make_dataset(DatasetConfig(n_per_class=5, seed=21))
    b = make_dataset(DatasetConfig(n_per_class=5, seed=21))
    np.testing.assert_array_equal(a.x, b.x)
    np.testing.assert_array_equal(a.y, b.y)


def test_dataset_cache_roundtrip(tmp_path: Path) -> None:
    config = DatasetConfig(n_per_class=4, seed=3)
    first = make_dataset(config, tmp_path)
    assert len(list(tmp_path.glob("pulse_synth_*.npz"))) == 1
    second = make_dataset(config, tmp_path)
    np.testing.assert_array_equal(first.x, second.x)
    loaded = Dataset.load(tmp_path / f"pulse_synth_{config.cache_key()}.npz")
    assert loaded.x.dtype == np.float32 and loaded.y.dtype == np.int64


def test_cache_key_depends_on_config() -> None:
    assert DatasetConfig(seed=1).cache_key() != DatasetConfig(seed=2).cache_key()
    assert DatasetConfig(n_per_class=4).cache_key() != DatasetConfig(n_per_class=5).cache_key()
