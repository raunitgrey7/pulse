"""Synthetic accelerometer windows for a motor-driven shaft on ball bearings.

Why synthetic: public bearing datasets are small, recorded at a handful of
fixed speeds and labelled per file, which makes it impossible to know how a
model behaves across speed, severity and noise. Here every window is drawn
from an explicit mechanical model, so the ground truth is exact and the
difficulty is controllable.

The model, component by component:

* Shaft speed wanders inside a window (a linear ramp plus slow hunting),
  and every periodic component is built from the integrated phase, so the
  harmonics and fault frequencies track the instantaneous speed.
* A healthy machine still shows residual 1x and 2x shaft harmonics.
* Imbalance is a 1x harmonic whose amplitude grows with the square of speed
  (centrifugal force is m r omega^2).
* Misalignment is dominated by 2x with weaker 1x and 3x components.
* Bearing defects produce impacts at BPFO, BPFI or 2 x BSF. Each impact
  rings the structure at its resonance (a damped sinusoid). Inner-race
  defects rotate through the load zone, so their impacts are modulated at
  the shaft frequency; ball defects spin with the cage and are modulated at
  the cage frequency; outer-race defects sit in the load zone and are not
  modulated. Impact spacing carries random slip jitter.
* Background: broadband excitation of the same resonance (so the resonance
  band is not a fault-only cue), mains hum at 50 Hz, white sensor noise set
  by a signal-to-noise ratio, then a 16-bit ADC with clipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from pulse.signals.bearing import BearingGeometry, fault_frequencies

CLASS_NAMES: tuple[str, ...] = (
    "healthy",
    "imbalance",
    "misalignment",
    "outer_race",
    "inner_race",
    "ball",
)

SPEED_REFERENCE_HZ = 30.0
"""Shaft frequency (1800 rpm) at which imbalance amplitude equals its nominal value."""


class Condition(IntEnum):
    """Machine condition, used as the class label."""

    HEALTHY = 0
    IMBALANCE = 1
    MISALIGNMENT = 2
    OUTER_RACE = 3
    INNER_RACE = 4
    BALL = 5

    @property
    def is_bearing_fault(self) -> bool:
        return self in (Condition.OUTER_RACE, Condition.INNER_RACE, Condition.BALL)


@dataclass(frozen=True)
class MachineSpec:
    """Fixed properties of the monitored machine and its sensor chain."""

    fs_hz: float = 12000.0
    n_samples: int = 2048
    bearing: BearingGeometry = field(default_factory=BearingGeometry)
    resonance_hz: float = 3200.0
    damping_ratio: float = 0.05
    mains_hz: float = 50.0
    adc_bits: int = 16
    full_scale_g: float = 20.0

    def __post_init__(self) -> None:
        if self.resonance_hz >= self.fs_hz / 2:
            raise ValueError("resonance must be below the Nyquist frequency")
        if not 0 < self.damping_ratio < 1:
            raise ValueError("damping ratio must be in (0, 1)")
        if self.n_samples < 64:
            raise ValueError("a window needs at least 64 samples")


@dataclass(frozen=True)
class WindowParams:
    """Operating point and condition for one window."""

    condition: Condition
    rpm: float
    severity: float
    snr_db: float
    speed_drift: float = 0.02
    load_modulation: float = 0.5
    slip: float = 0.01

    def __post_init__(self) -> None:
        if self.rpm <= 0:
            raise ValueError("rpm must be positive")
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError("severity must be in [0, 1]")
        if not 0.0 <= self.speed_drift < 0.5:
            raise ValueError("speed drift must be in [0, 0.5)")


@dataclass(frozen=True)
class GeneratedWindow:
    """A window plus the hidden state the tests use to check the physics."""

    signal: np.ndarray
    shaft_hz: np.ndarray
    impact_times_s: np.ndarray


def _shaft_speed_profile(spec: MachineSpec, params: WindowParams, rng: np.random.Generator) -> np.ndarray:
    """Instantaneous shaft frequency: nominal speed, a ramp and slow hunting."""
    t = np.arange(spec.n_samples) / spec.fs_hz
    duration = spec.n_samples / spec.fs_hz
    ramp_share = rng.uniform(0.0, 1.0)
    ramp = ramp_share * rng.choice((-1.0, 1.0)) * (t / duration - 0.5) * 2.0
    hunt_hz = rng.uniform(0.5, 3.0)
    hunt = (1.0 - ramp_share) * np.sin(2.0 * math.pi * hunt_hz * t + rng.uniform(0, 2 * math.pi))
    drift = params.speed_drift * (ramp + hunt)
    return params.rpm / 60.0 * (1.0 + drift)


def _phase(shaft_hz: np.ndarray, fs_hz: float, order: float = 1.0) -> np.ndarray:
    """Integrated phase in radians of a component at ``order`` times shaft speed."""
    return 2.0 * math.pi * order * np.cumsum(shaft_hz) / fs_hz


def _harmonics(
    shaft_hz: np.ndarray, fs_hz: float, amplitudes: dict[int, float], rng: np.random.Generator
) -> np.ndarray:
    """Sum of shaft harmonics with the given peak amplitudes and random phases."""
    out = np.zeros_like(shaft_hz)
    for order, amp in amplitudes.items():
        out += amp * np.sin(_phase(shaft_hz, fs_hz, order) + rng.uniform(0, 2 * math.pi))
    return out


def _speed_factor(shaft_hz: np.ndarray) -> float:
    """Centrifugal scaling (f / f_ref)^2 using the mean speed of the window."""
    return float((shaft_hz.mean() / SPEED_REFERENCE_HZ) ** 2)


def _resonance_kernel(spec: MachineSpec) -> np.ndarray:
    """Unit-peak impulse response of the structural resonance, truncated at 1 percent."""
    omega_n = 2.0 * math.pi * spec.resonance_hz
    omega_d = omega_n * math.sqrt(1.0 - spec.damping_ratio**2)
    tau = math.log(100.0) / (spec.damping_ratio * omega_n)
    n = max(8, math.ceil(tau * spec.fs_hz))
    t = np.arange(n) / spec.fs_hz
    h = np.exp(-spec.damping_ratio * omega_n * t) * np.sin(omega_d * t)
    return h / np.abs(h).max()


def _impact_times(fault_hz: np.ndarray, fs_hz: float, slip: float, rng: np.random.Generator) -> np.ndarray:
    """Impact instants for a time-varying fault frequency with slip jitter."""
    duration = fault_hz.size / fs_hz
    times: list[float] = []
    t = rng.uniform(0.0, 1.0 / float(fault_hz[0]))
    while t < duration:
        times.append(t)
        idx = min(int(t * fs_hz), fault_hz.size - 1)
        period = 1.0 / float(fault_hz[idx])
        t += period * max(0.2, 1.0 + slip * rng.standard_normal())
    return np.asarray(times)


def _impact_train(
    spec: MachineSpec,
    params: WindowParams,
    shaft_hz: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Resonance ringing from a train of modulated impacts at the fault frequency."""
    orders = fault_frequencies(spec.bearing, 1.0)
    if params.condition is Condition.OUTER_RACE:
        order, mod_order = orders.bpfo, 0.0
    elif params.condition is Condition.INNER_RACE:
        order, mod_order = orders.bpfi, 1.0
    else:
        order, mod_order = 2.0 * orders.bsf, orders.ftf
    times = _impact_times(order * shaft_hz, spec.fs_hz, params.slip, rng)
    idx = np.minimum((times * spec.fs_hz).round().astype(int), spec.n_samples - 1)
    peak_g = 0.1 + 0.9 * params.severity
    amps = peak_g * rng.uniform(0.8, 1.2, size=times.size)
    if mod_order > 0.0:
        mod_phase = _phase(shaft_hz, spec.fs_hz, mod_order)[idx] + rng.uniform(0, 2 * math.pi)
        depth = params.load_modulation
        amps *= (1.0 + depth * np.cos(mod_phase)) / (1.0 + depth)
    impulses = np.zeros(spec.n_samples)
    np.add.at(impulses, idx, amps)
    ringing = np.convolve(impulses, _resonance_kernel(spec))[: spec.n_samples]
    return ringing, times


def _deterministic_part(
    spec: MachineSpec, params: WindowParams, shaft_hz: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Everything except noise: harmonics, impacts and mains hum."""
    residual = {1: rng.uniform(0.03, 0.08), 2: rng.uniform(0.01, 0.03)}
    signal = _harmonics(shaft_hz, spec.fs_hz, residual, rng)
    impacts = np.asarray([])
    if params.condition is Condition.IMBALANCE:
        amp = (0.08 + 0.6 * params.severity) * _speed_factor(shaft_hz)
        signal += _harmonics(shaft_hz, spec.fs_hz, {1: amp}, rng)
    elif params.condition is Condition.MISALIGNMENT:
        amp = 0.08 + 0.5 * params.severity
        signal += _harmonics(shaft_hz, spec.fs_hz, {1: 0.5 * amp, 2: amp, 3: 0.3 * amp}, rng)
    elif params.condition.is_bearing_fault:
        ringing, impacts = _impact_train(spec, params, shaft_hz, rng)
        signal += ringing
    t = np.arange(spec.n_samples) / spec.fs_hz
    hum = rng.uniform(0.0, 0.02) * np.sin(2 * math.pi * spec.mains_hz * t + rng.uniform(0, 2 * math.pi))
    return signal + hum, impacts


def _background(spec: MachineSpec, rng: np.random.Generator) -> np.ndarray:
    """Broadband excitation of the resonance present on every machine."""
    white = rng.standard_normal(spec.n_samples)
    coloured = np.convolve(white, _resonance_kernel(spec))[: spec.n_samples]
    return 0.02 * coloured / (coloured.std() + 1e-12)


def _sensor_chain(spec: MachineSpec, clean: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add white sensor noise at the requested SNR, then quantise and clip like an ADC."""
    signal_power = float(np.mean(clean**2)) + 1e-12
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noisy = clean + math.sqrt(noise_power) * rng.standard_normal(clean.size)
    step = 2.0 * spec.full_scale_g / (2**spec.adc_bits)
    quantised = np.round(noisy / step) * step
    return np.clip(quantised, -spec.full_scale_g, spec.full_scale_g)


def generate_window(spec: MachineSpec, params: WindowParams, rng: np.random.Generator) -> GeneratedWindow:
    """Synthesise one accelerometer window in g for the given operating point."""
    shaft_hz = _shaft_speed_profile(spec, params, rng)
    deterministic, impacts = _deterministic_part(spec, params, shaft_hz, rng)
    clean = deterministic + _background(spec, rng)
    signal = _sensor_chain(spec, clean, params.snr_db, rng).astype(np.float32)
    return GeneratedWindow(signal=signal, shaft_hz=shaft_hz, impact_times_s=impacts)
