"""Benchmark harness pieces that can be tested without training."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pulse.bench.__main__ import build_parser, main
from pulse.bench.check import Rule, check, evaluate, load_rules
from pulse.bench.latency import measure_latency, summarise
from pulse.bench.memprobe import peak_rss_mb, probe
from pulse.bench.run import BenchConfig, render_table
from pulse.models.runtime import ModelKind
from pulse.signals.dataset import Dataset


def test_summarise_percentiles() -> None:
    stats = summarise([1_000_000 * i for i in range(1, 101)])
    assert stats.p50_ms == pytest.approx(50.5)
    assert stats.p95_ms == pytest.approx(95.05)
    assert stats.min_ms == 1.0 and stats.repeats == 100
    with pytest.raises(ValueError):
        summarise([])


def test_measure_latency_counts_calls_and_warmup() -> None:
    calls: list[int] = []
    stats = measure_latency(calls.append, warmup=3, repeats=10)
    assert calls[:3] == [0, 1, 2] and len(calls) == 13 and stats.repeats == 10
    assert stats.mean_ms >= stats.min_ms > 0.0


def _results(**overrides: float) -> dict[str, object]:
    row = {"accuracy": 0.95, "size_ratio_vs_fp32": 0.3}
    row.update(overrides)
    return {"variants": {"cnn_onnx_int8_static": row}}


def test_rules_evaluate_pass_fail_and_missing() -> None:
    rules = [
        Rule("cnn_onnx_int8_static", "accuracy", ">=", 0.9),
        Rule("cnn_onnx_int8_static", "size_ratio_vs_fp32", "<=", 0.25),
        Rule("cnn_onnx_int8_static", "latency_p50_ms", "<=", 1.0),
    ]
    outcomes = evaluate(_results(), rules)
    assert [o.passed for o in outcomes] == [True, False, False]
    assert outcomes[2].actual is None and "missing" in outcomes[2].line()
    assert outcomes[0].line().startswith("PASS")


def test_rule_rejects_unknown_operator() -> None:
    with pytest.raises(ValueError):
        Rule.from_dict({"variant": "x", "metric": "y", "op": "!=", "value": 1})


def test_check_reads_files_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "results.json").write_text(json.dumps(_results()))
    (tmp_path / "thresholds.json").write_text(
        json.dumps({"rules": [{"variant": "cnn_onnx_int8_static", "metric": "accuracy", "op": ">=", "value": 0.9}]})
    )
    assert check(tmp_path / "results.json", tmp_path / "thresholds.json") is True
    assert "1/1 threshold rules passed" in capsys.readouterr().out
    assert len(load_rules(tmp_path / "thresholds.json")) == 1


def test_main_check_only_exit_codes(tmp_path: Path) -> None:
    (tmp_path / "results.json").write_text(json.dumps(_results(accuracy=0.5)))
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps({"rules": [{"variant": "cnn_onnx_int8_static", "metric": "accuracy", "op": ">=", "value": 0.9}]})
    )
    assert main(["--check-only", "--out", str(tmp_path), "--thresholds", str(thresholds)]) == 1
    (tmp_path / "results.json").write_text(json.dumps(_results(accuracy=0.95)))
    assert main(["--check-only", "--out", str(tmp_path), "--thresholds", str(thresholds)]) == 0
    assert main(["--check-only", "--out", str(tmp_path), "--thresholds", str(tmp_path / "absent.json")]) == 0


def test_parser_defaults() -> None:
    args = build_parser().parse_args([])
    assert args.epochs == 20 and args.n_train == 400 and args.n_test == 150 and not args.check_only
    assert BenchConfig().inference_threads == 1 and BenchConfig().latency_windows == 300


def test_render_table_has_one_row_per_variant() -> None:
    row = {
        "accuracy": 0.9,
        "macro_f1": 0.89,
        "size_bytes": 2048,
        "latency_p50_ms": 0.1,
        "latency_p95_ms": 0.2,
        "peak_rss_mb": 50.0,
        "load_seconds": 0.01,
    }
    table = render_table({"a": row, "b": row})
    lines = table.splitlines()
    assert lines[0].startswith("| Variant |") and len(lines) == 4 and "| a | 0.9000 |" in lines[2]


def test_peak_rss_is_positive() -> None:
    assert peak_rss_mb() > 10.0


def test_probe_scores_every_window_in_a_fresh_predictor(tiny_onnx: Path, small_test: Dataset, tmp_path: Path) -> None:
    data = tmp_path / "test.npz"
    small_test.save(data)
    result = probe(ModelKind.ONNX, tiny_onnx, data, threads=1, name="tiny")
    assert result["n"] == len(small_test) and 0.0 <= result["accuracy"] <= 1.0
    assert result["peak_rss_mb"] > 10.0 and result["load_seconds"] > 0.0 and result["name"] == "tiny"
