"""The pulse command line."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from pulse import cli
from pulse.registry.releases import ReleaseManifest, verify_release
from pulse.registry.signing import SigningKey


def test_keygen_prints_a_usable_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["keygen"]) == 0
    out = json.loads(capsys.readouterr().out)
    key = SigningKey.from_seed(bytes.fromhex(out["seed_hex"]))
    assert key.verify_key().to_hex() == out["public_key_hex"]
    assert key.verify_key().fingerprint == out["fingerprint"]


def test_release_without_api_prints_a_verifiable_manifest(
    tmp_path: Path, signing_key: SigningKey, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"onnx" * 10)
    results = tmp_path / "results.json"
    results.write_text(
        json.dumps(
            {
                "variants": {
                    "cnn_onnx_int8_static": {
                        "accuracy": 0.91,
                        "macro_f1": 0.9,
                        "latency_p50_ms": 0.2,
                        "peak_rss_mb": 80.0,
                    }
                }
            }
        )
    )
    code = cli.main(
        [
            "release",
            "--model",
            str(model),
            "--version",
            "1.0.0",
            "--seed-hex",
            signing_key.to_seed().hex(),
            "--metrics-from",
            str(results),
        ]
    )
    assert code == 0
    manifest = ReleaseManifest.from_json(capsys.readouterr().out)
    assert verify_release(signing_key.verify_key(), manifest, model.read_bytes()).ok
    assert manifest.metrics["accuracy"] == 0.91 and manifest.model_name == "m.onnx"


def test_release_posts_to_api(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, signing_key: SigningKey, capsys: pytest.CaptureFixture[str]
) -> None:
    model = tmp_path / "m.onnx"
    model.write_bytes(b"x" * 20)
    captured: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> httpx.Response:
        captured.update(url=url, body=json, headers=headers)
        return httpx.Response(201, json={"ok": True}, request=httpx.Request("POST", url))

    monkeypatch.setattr(cli.httpx, "post", fake_post)
    code = cli.main(
        [
            "release",
            "--model",
            str(model),
            "--version",
            "2.0.0",
            "--seed-hex",
            signing_key.to_seed().hex(),
            "--api",
            "http://api",
            "--api-key",
            "k",
            "--metrics-from",
            str(tmp_path / "none.json"),
        ]
    )
    assert code == 0
    assert captured["url"] == "http://api/v1/registry/releases" and captured["headers"] == {"X-API-Key": "k"}
    assert captured["body"]["signer_public_key_hex"] == signing_key.verify_key().to_hex()
    assert json.loads(capsys.readouterr().out)["status"] == 201


def test_promote_and_rollback_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_post(url: str, json: Any, headers: dict[str, str], timeout: float) -> httpx.Response:
        calls.append((url, json))
        return httpx.Response(200, json={}, request=httpx.Request("POST", url))

    monkeypatch.setattr(cli.httpx, "post", fake_post)
    assert cli.main(["promote", "--version", "1.0.0", "--api", "http://api"]) == 0
    assert cli.main(["promote", "--version", "1.0.0", "--api", "http://api", "--rollback", "bad"]) == 0
    assert calls == [
        ("http://api/v1/registry/releases/1.0.0/promote", None),
        ("http://api/v1/registry/releases/1.0.0/rollback", {"reason": "bad"}),
    ]


def test_generate_writes_a_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "sample.npz"
    assert cli.main(["generate", "--out", str(out), "--n-per-class", "2", "--seed", "1"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["windows"] == 12
    with np.load(out) as f:
        assert f["x"].shape == (12, 2048)


def test_parser_requires_a_command() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_fleet_and_bench_arguments_pass_through(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, list[str]] = {}

    def fake_main(argv: list[str]) -> int:
        seen["argv"] = argv
        return 0

    monkeypatch.setattr("pulse.fleet.__main__.main", fake_main)
    assert cli.main(["fleet", "--api", "http://x", "--rounds", "3"]) == 0
    assert seen["argv"] == ["--api", "http://x", "--rounds", "3"]
    monkeypatch.setattr("pulse.bench.__main__.main", fake_main)
    assert cli.main(["bench", "--check-only"]) == 0
    assert seen["argv"] == ["--check-only"]
