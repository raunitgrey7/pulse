"""HTTP surface: health, metrics, auth, validation, ingest, alerts, registry."""

from __future__ import annotations

import base64
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pulse.api.app import create_app
from pulse.api.logging import JsonFormatter
from pulse.api.settings import Settings
from pulse.registry.releases import sign_release
from pulse.registry.signing import SigningKey
from tests.conftest import fault_report, make_report

MODEL = b"onnx" * 64


def _release_body(key: SigningKey, version: str = "1.0.0", model: bytes = MODEL) -> dict[str, object]:
    manifest = sign_release(key, version, "cnn.onnx", "onnx", model, {"accuracy": 0.93})
    return {
        "manifest": asdict(manifest),
        "model_b64": base64.b64encode(model).decode(),
        "signer_public_key_hex": key.verify_key().to_hex(),
    }


def _ingest(
    client: TestClient, reports: list[dict[str, object]], headers: dict[str, str] | None = None
) -> dict[str, object]:
    response = client.post("/v1/telemetry", json={"reports": reports}, headers=headers or {})
    assert response.status_code == 202, response.text
    return response.json()


def test_health_endpoint(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["devices"] == 0 and body["releases"] == 0


def test_dashboard_and_static_assets_are_served(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "Pulse fleet" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200


def test_metrics_are_prometheus_text(client: TestClient) -> None:
    client.get("/health")
    body = client.get("/metrics")
    assert body.headers["content-type"].startswith("text/plain")
    assert "# TYPE pulse_http_requests_total counter" in body.text
    assert 'pulse_http_requests_total{method="GET",path="/health",status="200"} 1' in body.text
    assert "pulse_http_request_duration_seconds_bucket" in body.text


def test_metrics_use_route_templates_not_raw_paths(client: TestClient) -> None:
    client.get("/v1/devices/does-not-exist")
    assert 'path="/v1/devices/{device_id}",status="404"' in client.get("/metrics").text


def test_request_id_is_echoed_or_generated(client: TestClient) -> None:
    assert client.get("/health", headers={"X-Request-ID": "trace-1"}).headers["X-Request-ID"] == "trace-1"
    generated = client.get("/health").headers["X-Request-ID"]
    assert len(generated) == 32


def test_api_key_is_required_when_configured(secured_client: TestClient) -> None:
    assert secured_client.get("/v1/devices").status_code == 401
    assert secured_client.get("/v1/devices", headers={"X-API-Key": "wrong"}).status_code == 401
    assert secured_client.get("/v1/devices", headers={"X-API-Key": "secret"}).status_code == 200
    assert secured_client.get("/health").status_code == 200
    assert secured_client.get("/metrics").status_code == 200


def test_ingest_accepts_valid_reports(client: TestClient) -> None:
    body = _ingest(client, [make_report(), make_report(device_id="dev-002")])
    assert body == {"accepted": 2, "devices_updated": 2, "alerts_opened": 0}
    devices = client.get("/v1/devices").json()
    assert [d["device_id"] for d in devices] == ["dev-001", "dev-002"]
    assert devices[0]["health"]["band"] == "healthy"


@pytest.mark.parametrize(
    "bad",
    [
        {"probabilities": [0.5] * 6},
        {"probabilities": [1.0]},
        {"predicted_class": "melted"},
        {"device_id": "dev 001; drop table"},
        {"rpm": -5.0},
        {"rms_g": 1e9},
        {"unexpected": 1},
        {"status": "weird"},
    ],
)
def test_ingest_rejects_malformed_reports(client: TestClient, bad: dict[str, object]) -> None:
    response = client.post("/v1/telemetry", json={"reports": [make_report(**bad)]})
    assert response.status_code == 422


def test_ingest_rejects_non_finite_numbers_with_a_clean_422(client: TestClient) -> None:
    raw = json.dumps({"reports": [make_report(rms_g=float("nan"))]})
    response = client.post("/v1/telemetry", content=raw, headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert response.json()["detail"][0]["msg"] == "Input should be a finite number"


def test_ingest_rejects_oversized_batches_and_empty_batches(client: TestClient) -> None:
    assert client.post("/v1/telemetry", json={"reports": []}).status_code == 422
    assert client.post("/v1/telemetry", json={"reports": [make_report()] * 501}).status_code == 422


def test_device_detail_and_baseline_fixing(client: TestClient) -> None:
    now = time.time()
    _ingest(client, [make_report(timestamp=now + i, rms_g=0.07 + 0.001 * i) for i in range(20)])
    detail = client.get("/v1/devices/dev-001?limit=5").json()
    assert detail["n_reports"] == 20 and len(detail["recent"]) == 5
    assert detail["baseline_rms_g"] == pytest.approx(0.0795, abs=1e-3)
    assert len(detail["recent_scores"]) == 20
    assert client.get("/v1/devices/nobody").status_code == 404


def test_fault_reports_open_an_alert_that_resolves_on_recovery(client: TestClient) -> None:
    now = time.time()
    _ingest(client, [make_report(timestamp=now + i) for i in range(20)])
    body = _ingest(client, [fault_report(timestamp=now + 20 + i) for i in range(6)])
    assert body["alerts_opened"] >= 1
    open_alerts = client.get("/v1/alerts?open=true").json()
    rules = {a["rule"] for a in open_alerts}
    assert "fault_persistent" in rules and "health_critical" in rules
    acked = client.post(f"/v1/alerts/{open_alerts[0]['id']}/ack").json()
    assert acked["acknowledged"] is True
    assert client.post("/v1/alerts/999/ack").status_code == 404
    _ingest(client, [make_report(timestamp=now + 40 + i) for i in range(12)])
    assert client.get("/v1/alerts?open=true").json() == []
    assert len(client.get("/v1/alerts?device_id=dev-001").json()) == len(open_alerts)


def test_fleet_summary_counts_bands_and_versions(client: TestClient) -> None:
    _ingest(client, [make_report(), make_report(device_id="dev-002", model_version="1.0.0")])
    summary = client.get("/v1/fleet/summary").json()
    assert summary["devices"] == 2 and summary["bands"]["healthy"] == 2
    assert summary["versions"] == {"builtin": 1, "1.0.0": 1} and summary["reports_total"] == 2
    assert 'pulse_devices{band="healthy"} 2' in client.get("/metrics").text


def test_stale_check_endpoint(client: TestClient) -> None:
    _ingest(client, [make_report(timestamp=time.time() - 3600)])
    assert client.post("/v1/alerts/check-stale").json() == {"opened": 1}
    assert client.get("/v1/alerts?open=true").json()[0]["rule"] == "device_stale"


def test_publish_get_and_download_release(client: TestClient, signing_key: SigningKey) -> None:
    created = client.post("/v1/registry/releases", json=_release_body(signing_key))
    assert created.status_code == 201
    assert created.json()["rollout"] == {
        "version": "1.0.0",
        "stage": "draft",
        "percent": 0,
        "history": created.json()["rollout"]["history"],
    }
    listed = client.get("/v1/registry/releases").json()
    assert len(listed) == 1 and listed[0]["cohort"]["n_reports"] == 0
    assert client.get("/v1/registry/releases/1.0.0/model").content == MODEL
    assert client.get("/v1/registry/releases/9.9.9").status_code == 404
    assert client.post("/v1/registry/releases", json=_release_body(signing_key)).status_code == 409


def test_tampered_or_unsigned_releases_are_refused(client: TestClient, signing_key: SigningKey) -> None:
    body = _release_body(signing_key)
    tampered = {**body, "model_b64": base64.b64encode(MODEL[:-1] + b"!").decode()}
    assert client.post("/v1/registry/releases", json=tampered).status_code == 403
    forged = {**body, "manifest": {**body["manifest"], "version": "2.0.0"}}  # type: ignore[dict-item]
    assert client.post("/v1/registry/releases", json=forged).status_code == 403
    assert client.post("/v1/registry/releases", json={**body, "model_b64": "%%%"}).status_code == 422
    assert client.get("/v1/registry/releases").json() == []


def test_trusted_key_pins_the_publisher(tmp_path: Path, signing_key: SigningKey) -> None:
    other = SigningKey.from_seed(bytes([9] * 32))
    settings = Settings(
        db_path=tmp_path / "p.db",
        registry_dir=tmp_path / "preg",
        verify_key_hex=signing_key.verify_key().to_hex(),
        log_level="WARNING",
    )
    with TestClient(create_app(settings)) as client:
        assert client.post("/v1/registry/releases", json=_release_body(other)).status_code == 403
        assert client.post("/v1/registry/releases", json=_release_body(signing_key)).status_code == 201


def test_rollout_ladder_through_the_api(client: TestClient, signing_key: SigningKey) -> None:
    client.post("/v1/registry/releases", json=_release_body(signing_key))
    assert client.post("/v1/registry/releases/1.0.0/promote").json()["rollout"]["stage"] == "canary"
    blocked = client.post("/v1/registry/releases/1.0.0/promote")
    assert blocked.status_code == 409 and "reports" in blocked.json()["detail"]
    now = time.time()
    _ingest(
        client, [make_report(device_id=f"dev-{i:03d}", timestamp=now + i, model_version="1.0.0") for i in range(60)]
    )
    assert client.post("/v1/registry/releases/1.0.0/promote").json()["rollout"]["stage"] == "pilot"
    assert client.post("/v1/registry/releases/1.0.0/promote").json()["rollout"]["stage"] == "fleet"
    assert client.post("/v1/registry/releases/1.0.0/promote").status_code == 400
    assignment = client.get("/v1/registry/assignment/dev-xyz").json()
    assert assignment["version"] == "1.0.0" and assignment["model_path"] == "/v1/registry/releases/1.0.0/model"
    rolled = client.post("/v1/registry/releases/1.0.0/rollback", json={"reason": "noise"}).json()
    assert rolled["rollout"]["stage"] == "rolled_back"
    assert client.get("/v1/registry/assignment/dev-xyz").json()["version"] is None
    assert 'pulse_registry_releases{stage="rolled_back"} 1' in client.get("/metrics").text


def test_alarm_storm_blocks_promotion(client: TestClient, signing_key: SigningKey) -> None:
    client.post("/v1/registry/releases", json=_release_body(signing_key, "1.0.0"))
    client.post("/v1/registry/releases", json=_release_body(signing_key, "1.1.0", MODEL + b"v2"))
    now = time.time()
    _ingest(
        client, [make_report(device_id=f"old-{i:03d}", timestamp=now + i, model_version="1.0.0") for i in range(60)]
    )
    for _ in range(3):
        assert client.post("/v1/registry/releases/1.0.0/promote").status_code == 200
    client.post("/v1/registry/releases/1.1.0/promote")
    _ingest(
        client, [fault_report(device_id=f"new-{i:03d}", timestamp=now + i, model_version="1.1.0") for i in range(60)]
    )
    blocked = client.post("/v1/registry/releases/1.1.0/promote")
    assert blocked.status_code == 409 and "fault rate" in blocked.json()["detail"]


def test_json_formatter_emits_parseable_lines_with_extras() -> None:
    record = logging.LogRecord("pulse.test", logging.INFO, __file__, 1, "hello %s", ("world",), None)
    record.device_id = "dev-9"
    line = json.loads(JsonFormatter().format(record))
    assert line["message"] == "hello world" and line["device_id"] == "dev-9"
    assert line["level"] == "INFO" and "request_id" in line and line["ts"].endswith("Z")
