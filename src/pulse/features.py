"""Hand-crafted vibration features for the classical baseline.

These are the quantities a vibration analyst computes by hand, in four
groups. Time statistics (RMS, crest factor, kurtosis and friends) capture
level and impulsiveness. Log band energies summarise the spectrum shape in a
scale-free way. Shaft-order amplitudes read the 1x, 2x and 3x harmonics at
the known speed, which separates imbalance from misalignment. Envelope
analysis band-passes the structural resonance, demodulates it with the
Hilbert transform and reads the envelope spectrum at the bearing fault
frequencies, which is how bearing defects are diagnosed in practice. The
extractor needs the nominal speed (from the tachometer) and the bearing
geometry, both known on a real machine.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

from pulse.signals.bearing import fault_frequencies
from pulse.signals.generator import MachineSpec

N_BANDS = 16
ORDER_TOLERANCE = 0.05
EPS = 1e-12


def _kurtosis(x: np.ndarray) -> float:
    """Pearson kurtosis (3.0 for a Gaussian)."""
    centred = x - x.mean()
    var = float(np.mean(centred**2)) + EPS
    return float(np.mean(centred**4) / var**2)


def _skewness(x: np.ndarray) -> float:
    centred = x - x.mean()
    std = float(np.sqrt(np.mean(centred**2))) + EPS
    return float(np.mean(centred**3) / std**3)


def _peak_near(freqs: np.ndarray, mag: np.ndarray, target_hz: float, tol: float) -> float:
    """Largest magnitude within a relative tolerance band around ``target_hz``."""
    lo, hi = target_hz * (1.0 - tol), target_hz * (1.0 + tol)
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        idx = int(np.argmin(np.abs(freqs - target_hz)))
        return float(mag[idx])
    return float(mag[mask].max())


class FeatureExtractor:
    """Turns a raw window plus its nominal speed into a fixed-length vector."""

    def __init__(self, spec: MachineSpec) -> None:
        self.spec = spec
        self.fs = spec.fs_hz
        self.n = spec.n_samples
        self._hann = np.hanning(self.n)
        self._freqs = np.fft.rfftfreq(self.n, 1.0 / self.fs)
        nyquist = self.fs / 2.0
        lo = spec.resonance_hz * 0.6
        hi = min(spec.resonance_hz * 1.4, 0.95 * nyquist)
        self._sos = sps.butter(4, [lo, hi], btype="bandpass", fs=self.fs, output="sos")
        self.names: list[str] = self._build_names()

    def _build_names(self) -> list[str]:
        time_names = [
            "rms",
            "peak",
            "crest_factor",
            "kurtosis",
            "skewness",
            "shape_factor",
            "impulse_factor",
            "clearance_factor",
            "zero_crossing_rate",
        ]
        band_names = [f"log_band_{i:02d}" for i in range(N_BANDS)]
        spectral_names = ["spectral_centroid_hz", "spectral_entropy", "spectral_flatness", "high_band_ratio"]
        order_names = ["order_1x", "order_2x", "order_3x", "log_2x_over_1x"]
        env_names = [
            "env_bpfo_1",
            "env_bpfo_2",
            "env_bpfi_1",
            "env_bpfi_2",
            "env_bsf_2",
            "env_bsf_4",
            "env_ftf",
            "env_shaft_1x",
            "env_bpfi_sidebands",
            "band_kurtosis",
        ]
        return time_names + band_names + spectral_names + order_names + env_names

    @property
    def n_features(self) -> int:
        return len(self.names)

    def _time_features(self, x: np.ndarray) -> list[float]:
        centred = x - x.mean()
        rms = float(np.sqrt(np.mean(centred**2))) + EPS
        peak = float(np.abs(centred).max())
        mean_abs = float(np.mean(np.abs(centred))) + EPS
        root_mean_sqrt = float(np.mean(np.sqrt(np.abs(centred)))) ** 2 + EPS
        zero_crossings = float(np.mean(np.signbit(centred[1:]) != np.signbit(centred[:-1])))
        return [
            rms,
            peak,
            peak / rms,
            _kurtosis(centred),
            _skewness(centred),
            rms / mean_abs,
            peak / mean_abs,
            peak / root_mean_sqrt,
            zero_crossings,
        ]

    def _spectrum(self, x: np.ndarray) -> np.ndarray:
        return np.abs(np.fft.rfft((x - x.mean()) * self._hann))

    def _spectral_features(self, mag: np.ndarray) -> list[float]:
        power = mag**2
        total = float(power.sum()) + EPS
        bands = np.array_split(power[: self.n // 2], N_BANDS)
        log_bands = [float(np.log10(b.sum() / total + EPS)) for b in bands]
        p = power / total
        centroid = float((self._freqs * p).sum())
        entropy = float(-(p * np.log(p + EPS)).sum() / np.log(p.size))
        flatness = float(np.exp(np.mean(np.log(power + EPS))) / (power.mean() + EPS))
        high = float(power[self._freqs >= 2000.0].sum() / total)
        return [*log_bands, centroid, entropy, flatness, high]

    def _order_features(self, mag: np.ndarray, shaft_hz: float) -> list[float]:
        scale = float(np.sqrt((mag**2).sum())) + EPS
        orders = [_peak_near(self._freqs, mag, k * shaft_hz, ORDER_TOLERANCE) / scale for k in (1, 2, 3)]
        return [*orders, float(np.log((orders[1] + EPS) / (orders[0] + EPS)))]

    def _envelope_features(self, x: np.ndarray, shaft_hz: float) -> list[float]:
        band = sps.sosfiltfilt(self._sos, x - x.mean())
        envelope = np.abs(sps.hilbert(band))
        env_mag = np.abs(np.fft.rfft((envelope - envelope.mean()) * self._hann))
        floor = float(np.median(env_mag[2:])) + EPS
        f = fault_frequencies(self.spec.bearing, shaft_hz)
        targets = [f.bpfo, 2 * f.bpfo, f.bpfi, 2 * f.bpfi, 2 * f.bsf, 4 * f.bsf, f.ftf, f.shaft_hz]
        peaks = [_peak_near(self._freqs, env_mag, t, ORDER_TOLERANCE) / floor for t in targets]
        sidebands = (
            0.5
            * (
                _peak_near(self._freqs, env_mag, f.bpfi - f.shaft_hz, ORDER_TOLERANCE)
                + _peak_near(self._freqs, env_mag, f.bpfi + f.shaft_hz, ORDER_TOLERANCE)
            )
            / floor
        )
        return [*peaks, sidebands, _kurtosis(band)]

    def extract(self, x: np.ndarray, rpm: float) -> np.ndarray:
        """Feature vector for one window given its nominal speed in rpm."""
        if x.shape != (self.n,):
            raise ValueError(f"expected a window of shape ({self.n},), got {x.shape}")
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        x64 = np.asarray(x, dtype=np.float64)
        shaft_hz = rpm / 60.0
        mag = self._spectrum(x64)
        values = (
            self._time_features(x64)
            + self._spectral_features(mag)
            + self._order_features(mag, shaft_hz)
            + self._envelope_features(x64, shaft_hz)
        )
        return np.asarray(values, dtype=np.float32)

    def extract_batch(self, x: np.ndarray, rpm: np.ndarray) -> np.ndarray:
        """Feature matrix for many windows."""
        return np.stack([self.extract(x[i], float(rpm[i])) for i in range(x.shape[0])])
