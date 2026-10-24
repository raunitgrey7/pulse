"""Health scoring and alert lifecycle."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from pulse.health.alerts import AlertManager, AlertRules, Candidate, Severity
from pulse.health.scoring import (
    HealthScore,
    HealthScorer,
    TelemetryPoint,
    band_for,
    ewma,
    impulsiveness_component,
    level_component,
    trend_component,
    vibration_zone,
)


def _points(n: int, rms: float, kurt: float, p_fault: float, cls: str = "healthy") -> list[TelemetryPoint]:
    return [TelemetryPoint(float(i), rms, kurt, p_fault, cls) for i in range(n)]


def test_ewma_weights_recent_values() -> None:
    assert ewma([1.0], 0.5) == 1.0
    assert ewma([0.0, 1.0], 0.5) == 0.5
    assert ewma([0.0, 0.0, 1.0], 0.9) > 0.85
    with pytest.raises(ValueError):
        ewma([], 0.3)


def test_vibration_zone_boundaries() -> None:
    assert vibration_zone(1.0) == "A" and vibration_zone(1.5) == "A"
    assert vibration_zone(2.0) == "B" and vibration_zone(3.0) == "C" and vibration_zone(4.1) == "D"


@settings(max_examples=40, deadline=None)
@given(st.floats(0.0, 10.0), st.floats(0.0, 10.0))
def test_level_component_is_monotone_and_bounded(a: float, b: float) -> None:
    lo, hi = sorted((a, b))
    assert 0.0 <= level_component(hi) <= level_component(lo) <= 1.0


def test_impulsiveness_component_bounds() -> None:
    assert impulsiveness_component(3.0) == 1.0
    assert impulsiveness_component(8.0) == 0.0
    assert 0.0 < impulsiveness_component(5.5) < 1.0


def test_trend_component_penalises_rising_level() -> None:
    assert trend_component([1.0, 1.0, 1.0], 1.0) == pytest.approx(1.0)
    assert trend_component([1.0, 1.5, 2.0], 1.0) == pytest.approx(0.0, abs=1e-9)
    assert trend_component([2.0, 1.5, 1.0], 1.0) == 1.0
    assert trend_component([1.0, 2.0], 1.0) == 1.0


def test_band_thresholds() -> None:
    assert band_for(70.0) == "healthy" and band_for(69.9) == "watch" and band_for(39.9) == "critical"


def test_healthy_telemetry_scores_high() -> None:
    score = HealthScorer().score(_points(10, 0.07, 2.9, 0.05), baseline_rms=0.07)
    assert score.band == "healthy" and score.score > 90 and score.zone == "A" and score.dominant_class == "healthy"


def test_faulty_telemetry_scores_low() -> None:
    score = HealthScorer().score(_points(10, 0.4, 9.0, 0.95, "outer_race"), baseline_rms=0.07)
    assert score.band == "critical" and score.zone == "D" and score.dominant_class == "outer_race"
    assert score.components["fault"] < 0.1 and score.components["level"] == 0.0


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.tuples(st.floats(0.0, 5.0), st.floats(1.0, 50.0), st.floats(0.0, 1.0)), min_size=1, max_size=12),
    st.floats(0.001, 2.0),
)
def test_score_is_always_within_bounds(raw: list[tuple[float, float, float]], baseline: float) -> None:
    points = [TelemetryPoint(float(i), r, k, p, "healthy") for i, (r, k, p) in enumerate(raw)]
    score = HealthScorer().score(points, baseline)
    assert 0.0 <= score.score <= 100.0
    assert score.band in {"healthy", "watch", "critical"}
    assert all(0.0 <= v <= 1.0 for v in score.components.values())


def test_scoring_requires_points() -> None:
    with pytest.raises(ValueError):
        HealthScorer().score([], 1.0)


def _scores(values: list[float], zone: str = "A") -> list[HealthScore]:
    return [HealthScore(v, band_for(v), {}, zone, "healthy") for v in values]


def test_health_critical_rule_needs_consecutive_lows() -> None:
    rules = AlertRules(critical_consecutive=3)
    assert rules.evaluate(_scores([30, 30]), [], []) == []
    assert rules.evaluate(_scores([30, 80, 30]), [], []) == []
    found = rules.evaluate(_scores([90, 30, 30, 30]), [], [])
    assert [c.rule for c in found] == ["health_critical"] and found[0].severity is Severity.CRITICAL


def test_fault_persistent_rule_requires_same_class_and_confidence() -> None:
    rules = AlertRules(fault_consecutive=3, fault_probability=0.8)
    assert rules.evaluate([], [0.9, 0.9, 0.9], ["ball", "ball", "outer_race"]) == []
    assert rules.evaluate([], [0.9, 0.5, 0.9], ["ball"] * 3) == []
    assert rules.evaluate([], [0.9, 0.9, 0.9], ["healthy"] * 3) == []
    found = rules.evaluate([], [0.9, 0.9, 0.9], ["ball"] * 3)
    assert found[0].rule == "fault_persistent" and found[0].details["predicted_class"] == "ball"


def test_zone_d_rule_fires_on_latest_score() -> None:
    found = AlertRules().evaluate(_scores([80], zone="D"), [], [])
    assert [c.rule for c in found] == ["vibration_zone_d"]


def test_alert_manager_dedupes_resolves_and_acknowledges() -> None:
    clock = {"t": 100.0}
    seen: list[int] = []
    manager = AlertManager(sinks=[lambda a: seen.append(a.id)], clock=lambda: clock["t"])
    candidate = Candidate("health_critical", Severity.CRITICAL, "low", {})
    opened = manager.apply("dev-1", [candidate])
    assert len(opened) == 1 and seen == [1]
    clock["t"] = 101.0
    assert manager.apply("dev-1", [candidate]) == []
    alert = manager.list(open_only=True)[0]
    assert alert.occurrences == 2 and alert.updated_at == 101.0 and seen == [1]
    manager.acknowledge(alert.id)
    assert manager.get(alert.id).acknowledged
    manager.apply("dev-1", [])
    assert manager.list(open_only=True) == []
    assert manager.list()[0].resolved_at == 101.0


def test_alert_manager_isolates_devices() -> None:
    manager = AlertManager(sinks=[])
    candidate = Candidate("fault_persistent", Severity.WARNING, "x", {})
    manager.apply("a", [candidate])
    manager.apply("b", [candidate])
    manager.apply("a", [])
    assert [x.device_id for x in manager.list(open_only=True)] == ["b"]
    assert len(manager.list(device_id="a")) == 1


def test_stale_alert_opens_and_clears() -> None:
    clock = {"t": 1000.0}
    manager = AlertManager(sinks=[], clock=lambda: clock["t"])
    assert manager.stale("dev-1", last_seen=990.0, stale_after_s=60.0) is None
    clock["t"] = 1100.0
    alert = manager.stale("dev-1", last_seen=990.0, stale_after_s=60.0)
    assert alert is not None and alert.rule == "device_stale"
    assert manager.stale("dev-1", last_seen=990.0, stale_after_s=60.0) is None
    assert manager.list(open_only=True)[0].occurrences == 2
    manager.stale("dev-1", last_seen=1099.0, stale_after_s=60.0)
    assert manager.list(open_only=True) == []
