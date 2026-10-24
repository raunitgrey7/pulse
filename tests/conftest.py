"""Shared fixtures: small datasets, a quickly trained CNN exported to ONNX, API clients."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pulse.api.app import create_app
from pulse.api.settings import Settings
from pulse.models.cnn import PulseCNN, TrainConfig, train_cnn
from pulse.models.export import export_onnx
from pulse.registry.signing import SigningKey
from pulse.signals.dataset import Dataset, DatasetConfig, make_dataset
from pulse.signals.generator import MachineSpec


@pytest.fixture(scope="session")
def spec() -> MachineSpec:
    return MachineSpec()


@pytest.fixture(scope="session")
def small_train() -> Dataset:
    return make_dataset(DatasetConfig(n_per_class=30, seed=1))


@pytest.fixture(scope="session")
def small_test() -> Dataset:
    return make_dataset(DatasetConfig(n_per_class=15, seed=2))


@pytest.fixture(scope="session")
def tiny_model(small_train: Dataset) -> PulseCNN:
    """A CNN trained for two epochs: enough for logits that mean something, fast enough for tests."""
    model, _ = train_cnn(small_train, TrainConfig(epochs=2, threads=2, seed=0, val_fraction=0.0))
    return model


@pytest.fixture(scope="session")
def tiny_onnx(tiny_model: PulseCNN, tmp_path_factory: pytest.TempPathFactory, spec: MachineSpec) -> Path:
    path = tmp_path_factory.mktemp("models") / "tiny.onnx"
    export_onnx(tiny_model, path, spec.n_samples)
    return path


@pytest.fixture(scope="session")
def signing_key() -> SigningKey:
    return SigningKey.from_seed(bytes(range(32)))


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "pulse.db", registry_dir=tmp_path / "registry", log_level="WARNING")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


@pytest.fixture
def secured_client(tmp_path: Path) -> Iterator[TestClient]:
    s = Settings(api_key="secret", db_path=tmp_path / "s.db", registry_dir=tmp_path / "sreg", log_level="WARNING")
    with TestClient(create_app(s)) as c:
        yield c


def make_report(**overrides: Any) -> dict[str, Any]:
    """A valid healthy telemetry report with optional field overrides."""
    report: dict[str, Any] = {
        "device_id": "dev-001",
        "timestamp": time.time(),
        "rpm": 1800.0,
        "rms_g": 0.07,
        "kurtosis": 2.9,
        "crest_factor": 3.4,
        "predicted_class": "healthy",
        "probabilities": [0.9, 0.02, 0.02, 0.02, 0.02, 0.02],
        "model_version": "builtin",
        "latency_ms": 0.4,
    }
    report.update(overrides)
    return report


def fault_report(**overrides: Any) -> dict[str, Any]:
    base = make_report(
        predicted_class="outer_race",
        probabilities=[0.04, 0.01, 0.01, 0.9, 0.02, 0.02],
        rms_g=0.45,
        kurtosis=9.0,
        crest_factor=8.0,
    )
    base.update(overrides)
    return base
