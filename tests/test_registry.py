"""Signing, release verification and staged rollout."""

from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pulse.registry.releases import ReleaseManifest, parse_version, sign_release, verify_release
from pulse.registry.rollout import STAGE_PERCENT, CanaryGuard, CohortStats, ModelRegistry, Stage, device_bucket
from pulse.registry.signing import SigningKey, VerifyKey, sha256_hex

MODEL = b"\x00onnx-model-bytes" * 100


def _manifest(key: SigningKey, version: str = "1.0.0", model: bytes = MODEL) -> ReleaseManifest:
    return sign_release(
        key, version, "cnn.onnx", "onnx", model, {"accuracy": 0.9}, "notes", created_at="2026-10-25T00:00:00+00:00"
    )


def test_sign_then_verify_roundtrip(signing_key: SigningKey) -> None:
    manifest = _manifest(signing_key)
    result = verify_release(signing_key.verify_key(), manifest, MODEL)
    assert result.ok and result.reason == "ok"
    assert manifest.sha256 == sha256_hex(MODEL) and manifest.size_bytes == len(MODEL)


def test_tampered_model_bytes_are_rejected(signing_key: SigningKey) -> None:
    manifest = _manifest(signing_key)
    result = verify_release(signing_key.verify_key(), manifest, MODEL[:-1] + b"\x01")
    assert not result.ok and "hash" in result.reason


def test_tampered_manifest_field_is_rejected(signing_key: SigningKey) -> None:
    manifest = replace(_manifest(signing_key), metrics={"accuracy": 0.999})
    result = verify_release(signing_key.verify_key(), manifest, MODEL)
    assert not result.ok and "signature" in result.reason


def test_release_from_another_key_is_rejected(signing_key: SigningKey) -> None:
    other = SigningKey.from_seed(bytes([7] * 32))
    manifest = _manifest(other)
    assert not verify_release(signing_key.verify_key(), manifest, MODEL).ok


def test_garbage_signature_hex_is_rejected(signing_key: SigningKey) -> None:
    manifest = replace(_manifest(signing_key), signature_hex="zz")
    assert "hex" in verify_release(signing_key.verify_key(), manifest, MODEL).reason


def test_manifest_json_roundtrip_preserves_signature(signing_key: SigningKey) -> None:
    manifest = _manifest(signing_key)
    again = ReleaseManifest.from_json(manifest.to_json())
    assert again == manifest
    assert verify_release(signing_key.verify_key(), again, MODEL).ok


def test_signing_is_deterministic_per_seed() -> None:
    a = SigningKey.from_seed(bytes(range(32)))
    b = SigningKey.from_seed(bytes(range(32)))
    assert a.verify_key() == b.verify_key()
    assert a.sign(b"x") == b.sign(b"x")
    assert SigningKey.generate().verify_key() != a.verify_key()


def test_seed_and_hex_roundtrips() -> None:
    key = SigningKey.generate()
    assert SigningKey.from_seed(key.to_seed()).verify_key() == key.verify_key()
    assert VerifyKey.from_hex(key.verify_key().to_hex()) == key.verify_key()
    assert len(key.verify_key().fingerprint) == 16


def test_verify_key_length_is_checked() -> None:
    with pytest.raises(ValueError):
        VerifyKey(b"short")
    with pytest.raises(ValueError):
        SigningKey.from_seed(b"short")


@pytest.mark.parametrize("bad", ["1.0", "v1.0.0", "1.0.0-rc1", "01.0.0a", "", "1..0"])
def test_versions_must_be_strict_semver(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_version(bad)


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 99), st.integers(0, 99), st.integers(0, 99))
def test_valid_versions_parse_to_ordered_tuples(a: int, b: int, c: int) -> None:
    assert parse_version(f"{a}.{b}.{c}") == (a, b, c)


@settings(max_examples=30, deadline=None)
@given(st.text(min_size=1, max_size=40))
def test_device_bucket_is_stable_and_in_range(device_id: str) -> None:
    assert 0 <= device_bucket(device_id) < 100
    assert device_bucket(device_id) == device_bucket(device_id)


def test_device_bucket_spreads_devices_roughly_uniformly() -> None:
    canaries = sum(device_bucket(f"dev-{i:05d}") < 5 for i in range(4000))
    assert 120 < canaries < 280


def test_canary_guard_decisions() -> None:
    guard = CanaryGuard(min_reports=10, max_error_rate=0.05, max_fault_rate_ratio=2.0)
    incumbent = CohortStats("1.0.0", 10, 500, 0.10, 0.0)
    assert not guard.decide(CohortStats("1.1.0", 1, 5, 0.1, 0.0), incumbent).ok
    assert "error rate" in guard.decide(CohortStats("1.1.0", 2, 50, 0.1, 0.2), incumbent).reason
    assert "fault rate" in guard.decide(CohortStats("1.1.0", 2, 50, 0.5, 0.0), incumbent).reason
    assert guard.decide(CohortStats("1.1.0", 2, 50, 0.15, 0.0), incumbent).ok
    assert guard.decide(CohortStats("1.1.0", 2, 50, 0.15, 0.0), None).ok


def test_canary_guard_uses_floor_when_incumbent_is_quiet() -> None:
    guard = CanaryGuard(min_reports=10, min_fault_rate_floor=0.05)
    incumbent = CohortStats("1.0.0", 10, 500, 0.0, 0.0)
    assert guard.decide(CohortStats("1.1.0", 2, 50, 0.09, 0.0), incumbent).ok
    assert not guard.decide(CohortStats("1.1.0", 2, 50, 0.11, 0.0), incumbent).ok


@pytest.fixture
def registry(tmp_path: Path) -> ModelRegistry:
    return ModelRegistry(tmp_path / "registry")


def test_publish_starts_as_draft_and_is_immutable(registry: ModelRegistry, signing_key: SigningKey) -> None:
    state = registry.publish(_manifest(signing_key), MODEL)
    assert state.stage is Stage.DRAFT and state.percent == 0
    assert registry.model_bytes("1.0.0") == MODEL
    assert registry.manifest("1.0.0") == _manifest(signing_key)
    with pytest.raises(FileExistsError):
        registry.publish(_manifest(signing_key), MODEL)


def test_versions_are_sorted_numerically(registry: ModelRegistry, signing_key: SigningKey) -> None:
    for v in ("1.9.0", "1.10.0", "0.1.0"):
        registry.publish(_manifest(signing_key, v), MODEL)
    assert registry.versions() == ["1.10.0", "1.9.0", "0.1.0"]


def test_ladder_requires_evidence_past_canary(registry: ModelRegistry, signing_key: SigningKey) -> None:
    registry.publish(_manifest(signing_key), MODEL)
    assert registry.promote("1.0.0").stage is Stage.CANARY
    with pytest.raises(ValueError):
        registry.promote("1.0.0")
    weak = CohortStats("1.0.0", 1, 3, 0.0, 0.0)
    with pytest.raises(PermissionError):
        registry.promote("1.0.0", CanaryGuard(min_reports=50), weak, None)
    strong = CohortStats("1.0.0", 5, 200, 0.05, 0.0)
    assert registry.promote("1.0.0", CanaryGuard(), strong, None).stage is Stage.PILOT
    assert registry.promote("1.0.0", CanaryGuard(), strong, None).stage is Stage.FLEET
    with pytest.raises(ValueError):
        registry.promote("1.0.0", CanaryGuard(), strong, None)


def test_rollback_removes_release_from_assignment(registry: ModelRegistry, signing_key: SigningKey) -> None:
    registry.publish(_manifest(signing_key), MODEL)
    for _ in range(3):
        registry.promote("1.0.0", CanaryGuard(min_reports=0), CohortStats("1.0.0", 1, 1, 0.0, 0.0), None)
    assert registry.assignment("any-device") == "1.0.0"
    state = registry.rollback("1.0.0", "false alarms")
    assert state.stage is Stage.ROLLED_BACK and state.percent == 0
    assert registry.assignment("any-device") is None
    assert state.history[-1]["reason"] == "false alarms"


def test_assignment_prefers_newest_covering_release(registry: ModelRegistry, signing_key: SigningKey) -> None:
    registry.publish(_manifest(signing_key, "1.0.0"), MODEL)
    registry.publish(_manifest(signing_key, "1.1.0"), MODEL + b"!")
    ok = CohortStats("x", 9, 999, 0.0, 0.0)
    for _ in range(3):
        registry.promote("1.0.0", CanaryGuard(), ok, None)
    registry.promote("1.1.0")
    canary = next(f"dev-{i}" for i in range(1000) if device_bucket(f"dev-{i}") < STAGE_PERCENT[Stage.CANARY])
    steady = next(f"dev-{i}" for i in range(1000) if device_bucket(f"dev-{i}") >= STAGE_PERCENT[Stage.PILOT])
    assert registry.assignment(canary) == "1.1.0"
    assert registry.assignment(steady) == "1.0.0"
    assert registry.incumbent() == "1.0.0"
    assert registry.incumbent(exclude="1.0.0") is None


def test_registry_state_survives_reopen(tmp_path: Path, signing_key: SigningKey) -> None:
    first = ModelRegistry(tmp_path / "r")
    first.publish(_manifest(signing_key), MODEL)
    first.promote("1.0.0")
    second = ModelRegistry(tmp_path / "r")
    assert second.state("1.0.0").stage is Stage.CANARY
    assert asdict(second.manifest("1.0.0")) == asdict(first.manifest("1.0.0"))
