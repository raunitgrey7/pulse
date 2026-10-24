"""Balanced labelled datasets drawn from the generator, cached under ``data/``.

Each window gets its own operating point: speed, fault severity and SNR are
drawn independently from the configured ranges, so a model has to generalise
across the whole operating envelope instead of memorising a fixed speed.
Train and test sets use different seeds and are therefore independent draws
from the same distribution. The cache key hashes the configuration and a
generator version, so a change to the physics invalidates stale files.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from pulse.signals.generator import CLASS_NAMES, Condition, MachineSpec, WindowParams, generate_window

GENERATOR_VERSION = 1


@dataclass(frozen=True)
class DatasetConfig:
    """What to draw and from which ranges."""

    n_per_class: int = 400
    seed: int = 7
    rpm_range: tuple[float, float] = (900.0, 3600.0)
    severity_range: tuple[float, float] = (0.05, 1.0)
    snr_db_range: tuple[float, float] = (-6.0, 15.0)
    spec: MachineSpec = field(default_factory=MachineSpec)

    def cache_key(self) -> str:
        payload = json.dumps({"v": GENERATOR_VERSION, **asdict(self)}, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class Dataset:
    """Windows with labels and the operating point of each window."""

    x: np.ndarray
    y: np.ndarray
    rpm: np.ndarray
    severity: np.ndarray
    snr_db: np.ndarray

    def __len__(self) -> int:
        return int(self.y.shape[0])

    @property
    def class_names(self) -> tuple[str, ...]:
        return CLASS_NAMES

    def save(self, path: Path) -> None:
        np.savez_compressed(path, x=self.x, y=self.y, rpm=self.rpm, severity=self.severity, snr_db=self.snr_db)

    @classmethod
    def load(cls, path: Path) -> Dataset:
        with np.load(path) as f:
            return cls(x=f["x"], y=f["y"], rpm=f["rpm"], severity=f["severity"], snr_db=f["snr_db"])


def _draw(config: DatasetConfig) -> Dataset:
    rng = np.random.default_rng(config.seed)
    n_total = config.n_per_class * len(Condition)
    x = np.empty((n_total, config.spec.n_samples), dtype=np.float32)
    y = np.repeat(np.arange(len(Condition), dtype=np.int64), config.n_per_class)
    rpm = rng.uniform(*config.rpm_range, size=n_total).astype(np.float32)
    severity = rng.uniform(*config.severity_range, size=n_total).astype(np.float32)
    snr_db = rng.uniform(*config.snr_db_range, size=n_total).astype(np.float32)
    for i in range(n_total):
        params = WindowParams(Condition(int(y[i])), float(rpm[i]), float(severity[i]), float(snr_db[i]))
        x[i] = generate_window(config.spec, params, rng).signal
    order = rng.permutation(n_total)
    return Dataset(x[order], y[order], rpm[order], severity[order], snr_db[order])


def make_dataset(config: DatasetConfig, cache_dir: Path | None = None) -> Dataset:
    """Draw the dataset, or load it from the cache if the same config was drawn before."""
    if cache_dir is None:
        return _draw(config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"pulse_synth_{config.cache_key()}.npz"
    if path.exists():
        return Dataset.load(path)
    dataset = _draw(config)
    dataset.save(path)
    return dataset
