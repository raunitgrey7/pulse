"""Device simulator and fleet runner, including model sync against the real API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pulse.api.schemas import TelemetryReport
from pulse.fleet.device import DegradationPlan, DeviceConfig, SimulatedDevice
from pulse.fleet.runner import FleetConfig, FleetRunner, RegistryClient, build_fleet
from pulse.registry.releases import ReleaseManifest, sign_release
from pulse.registry.signing import SigningKey
from pulse.signals.generator import CLASS_NAMES, Condition, MachineSpec


class StubSource:
    """In-memory model source so device sync can be tested without HTTP."""

    def __init__(self, version: str | None, manifest: ReleaseManifest | None, model: bytes) -> None:
        self.version, self.manifest, self.model = version, manifest, model
        self.fetches = 0

    def assignment(self, device_id: str) -> str | None:
        return self.version

    def fetch(self, version: str) -> tuple[ReleaseManifest, bytes]:
        self.fetches += 1
        assert self.manifest is not None
        return self.manifest, self.model


def _device(
    spec: MachineSpec, model: Path | None, key: SigningKey | None, plan: DegradationPlan | None = None
) -> SimulatedDevice:
    config = DeviceConfig("dev-t", 1800.0, 10.0, plan or DegradationPlan(), seed=1)
    return SimulatedDevice(config, spec, key.verify_key() if key else None, model)


def test_degradation_plan_ramps_after_onset() -> None:
    plan = DegradationPlan(Condition.BALL, onset_step=10, ramp_steps=10, final_severity=0.9)
    assert plan.at(0) == (Condition.HEALTHY, 0.0)
    assert plan.at(10) == (Condition.BALL, pytest.approx(0.05))
    assert plan.at(15)[1] == pytest.approx(0.475)
    assert plan.at(50) == (Condition.BALL, 0.9)
    assert DegradationPlan().at(1000) == (Condition.HEALTHY, 0.0)


def test_build_fleet_is_deterministic_and_respects_fault_fraction() -> None:
    a = build_fleet(FleetConfig(n_devices=8, fault_fraction=0.5, seed=3, rounds=40))
    b = build_fleet(FleetConfig(n_devices=8, fault_fraction=0.5, seed=3, rounds=40))
    assert [d.config for d in a] == [d.config for d in b]
    assert sum(d.config.plan.fault is not None for d in a) == 4
    assert all(10 <= d.config.plan.onset_step < 20 for d in a if d.config.plan.fault is not None)
    assert len({d.device_id for d in a}) == 8


def test_device_step_produces_a_valid_report(spec: MachineSpec, tiny_onnx: Path) -> None:
    device = _device(spec, tiny_onnx, None)
    report = device.step(timestamp=1.0)
    assert isinstance(report, TelemetryReport)
    assert report.status == "ok" and report.predicted_class in CLASS_NAMES
    assert abs(sum(report.probabilities) - 1.0) < 1e-5 and report.latency_ms > 0
    assert report.model_version == "builtin" and device.step_index == 1


def test_device_without_model_reports_error_status(spec: MachineSpec) -> None:
    report = _device(spec, None, None).step(timestamp=1.0)
    assert report.status == "error" and report.probabilities == pytest.approx([1 / 6] * 6)


def test_bearing_degradation_raises_kurtosis(spec: MachineSpec, tiny_onnx: Path) -> None:
    plan = DegradationPlan(Condition.OUTER_RACE, onset_step=5, ramp_steps=1, final_severity=1.0)
    device = _device(spec, tiny_onnx, None, plan)
    healthy = [device.step(float(i + 1)).kurtosis for i in range(5)]
    faulty = [device.step(float(i + 1)).kurtosis for i in range(5, 10)]
    assert max(healthy) < min(faulty)


def test_device_adopts_a_verified_release(spec: MachineSpec, tiny_onnx: Path, signing_key: SigningKey) -> None:
    model = tiny_onnx.read_bytes()
    manifest = sign_release(signing_key, "1.2.0", "tiny.onnx", "onnx", model)
    device = _device(spec, None, signing_key)
    assert device.sync_model(StubSource("1.2.0", manifest, model)) is True
    assert device.model_version == "1.2.0" and device.stats.model_updates == 1
    assert device.step(1.0).status == "ok"
    assert device.sync_model(StubSource("1.2.0", manifest, model)) is False


def test_device_rejects_tampered_release_and_does_not_retry(
    spec: MachineSpec, tiny_onnx: Path, signing_key: SigningKey
) -> None:
    model = tiny_onnx.read_bytes()
    manifest = sign_release(signing_key, "1.3.0", "tiny.onnx", "onnx", model)
    source = StubSource("1.3.0", manifest, model[:-10] + b"\x00" * 10)
    device = _device(spec, tiny_onnx, signing_key)
    assert device.sync_model(source) is False
    assert device.model_version == "builtin" and device.stats.rejected_releases == ["1.3.0"]
    assert device.sync_model(source) is False and source.fetches == 1


def test_device_without_trust_key_refuses_updates(spec: MachineSpec, tiny_onnx: Path, signing_key: SigningKey) -> None:
    model = tiny_onnx.read_bytes()
    manifest = sign_release(signing_key, "1.4.0", "tiny.onnx", "onnx", model)
    device = _device(spec, tiny_onnx, None)
    assert device.sync_model(StubSource("1.4.0", manifest, model)) is False
    assert device.stats.rejected_releases == ["1.4.0"]


def test_fleet_runner_streams_to_the_api_and_picks_up_canary_release(
    client: TestClient, tiny_onnx: Path, signing_key: SigningKey
) -> None:
    import base64
    from dataclasses import asdict

    model = tiny_onnx.read_bytes()
    manifest = sign_release(signing_key, "2.0.0", "tiny.onnx", "onnx", model)
    body = {
        "manifest": asdict(manifest),
        "model_b64": base64.b64encode(model).decode(),
        "signer_public_key_hex": signing_key.verify_key().to_hex(),
    }
    assert client.post("/v1/registry/releases", json=body).status_code == 201
    for _ in range(3):
        client.post("/v1/registry/releases/2.0.0/promote")
        if client.get("/v1/registry/releases/2.0.0").json()["rollout"]["stage"] == "canary":
            break
    config = FleetConfig(
        n_devices=4,
        fault_fraction=0.5,
        seed=1,
        rounds=6,
        interval_s=0.0,
        sync_every=3,
        fallback_model=tiny_onnx,
        trust_key_hex=signing_key.verify_key().to_hex(),
    )
    for _ in range(2):
        client.post("/v1/registry/releases/2.0.0/promote") if False else None
    devices = build_fleet(config)
    clock = iter(float(i) for i in range(1, 100))
    stats = FleetRunner(devices, client, config, clock=lambda: next(clock), sleep=lambda _: None).run()
    assert stats.rounds == 6 and stats.reports_sent == 24 and stats.reports_accepted == 24 and stats.post_failures == 0
    listed = client.get("/v1/devices").json()
    assert [d["device_id"] for d in listed] == [d.device_id for d in devices]
    canaries = [d for d in devices if d.model_version == "2.0.0"]
    assert len(canaries) == stats.model_updates
    summary = client.get("/v1/fleet/summary").json()
    assert summary["reports_total"] == 24


def test_registry_client_reads_assignment_and_model(client: TestClient, signing_key: SigningKey) -> None:
    import base64
    from dataclasses import asdict

    model = b"model-bytes" * 10
    manifest = sign_release(signing_key, "3.0.0", "m.onnx", "onnx", model)
    body = {
        "manifest": asdict(manifest),
        "model_b64": base64.b64encode(model).decode(),
        "signer_public_key_hex": signing_key.verify_key().to_hex(),
    }
    client.post("/v1/registry/releases", json=body)
    source = RegistryClient(client)
    assert source.assignment("dev-1") is None
    fetched_manifest, fetched_model = source.fetch("3.0.0")
    assert fetched_manifest == manifest and fetched_model == model
