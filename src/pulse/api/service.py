"""The ingest pipeline and the shared service objects behind the routes.

One ``Services`` instance per application holds the store, registry, scorer,
alert manager and metrics. ``ingest`` is the only write path for telemetry:
store the report, score the device on its recent history, persist the
score, run the alert rules and update metrics, in that order, so a failure
in alerting can never lose a report.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import httpx

from pulse.api.metrics import MetricsRegistry
from pulse.api.schemas import TelemetryReport
from pulse.api.settings import Settings
from pulse.api.store import TelemetryStore
from pulse.health.alerts import Alert, AlertManager, AlertRules, log_sink
from pulse.health.scoring import HealthScorer
from pulse.registry.rollout import CanaryGuard, ModelRegistry
from pulse.registry.signing import VerifyKey

SCORE_HISTORY = 10
RULE_HISTORY = 10
log = logging.getLogger("pulse.service")


def webhook_sink(url: str, timeout_s: float = 2.0) -> object:
    """Post each new alert as JSON; failures are logged, never raised into the request."""

    def sink(alert: Alert) -> None:
        try:
            httpx.post(url, json=alert.to_dict(), timeout=timeout_s)
        except httpx.HTTPError as exc:
            log.warning("alert webhook failed", extra={"error": str(exc), "alert_id": alert.id})

    return sink


@dataclass
class IngestOutcome:
    accepted: int = 0
    devices: set[str] = field(default_factory=set)
    alerts_opened: int = 0


class Services:
    """Everything a request handler needs, built once at startup."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.started_at = time.time()
        self.store = TelemetryStore(settings.db_path)
        self.registry = ModelRegistry(settings.registry_dir)
        self.guard = CanaryGuard()
        self.scorer = HealthScorer()
        self.rules = AlertRules(stale_after_s=settings.stale_after_s)
        sinks = [log_sink]
        if settings.alert_webhook_url:
            sinks.append(webhook_sink(settings.alert_webhook_url))  # type: ignore[arg-type]
        self.alerts = AlertManager(sinks)
        self.metrics = MetricsRegistry()
        self.trusted_key = VerifyKey.from_hex(settings.verify_key_hex) if settings.verify_key_hex else None

    def close(self) -> None:
        self.store.close()

    def ingest(self, reports: list[TelemetryReport]) -> IngestOutcome:
        outcome = IngestOutcome()
        for report in reports:
            self._ingest_one(report, outcome)
        self.refresh_gauges()
        return outcome

    def _ingest_one(self, report: TelemetryReport, outcome: IngestOutcome) -> None:
        device = self.store.insert_report(report)
        baseline = (
            device.baseline_rms if device.baseline_rms is not None else self.store.early_baseline(report.device_id)
        )
        points = self.store.recent_points(report.device_id, SCORE_HISTORY)
        score = self.scorer.score(points, baseline)
        self.store.insert_score(report.device_id, report.timestamp, score)
        scores = self.store.recent_scores(report.device_id, RULE_HISTORY)
        candidates = self.rules.evaluate(
            scores, [p.fault_probability for p in points], [p.predicted_class for p in points]
        )
        opened = self.alerts.apply(report.device_id, candidates)
        for alert in opened:
            self.metrics.alerts_opened.inc(severity=alert.severity.value, rule=alert.rule)
        self.metrics.telemetry_reports.inc(predicted_class=report.predicted_class)
        outcome.accepted += 1
        outcome.devices.add(report.device_id)
        outcome.alerts_opened += len(opened)

    def check_staleness(self) -> int:
        """Open staleness alerts for quiet devices; returns how many were opened."""
        opened = 0
        for device in self.store.devices():
            if self.alerts.stale(device.device_id, device.last_seen, self.settings.stale_after_s) is not None:
                opened += 1
        return opened

    def refresh_gauges(self) -> None:
        latest = self.store.latest_scores()
        bands = {"healthy": 0, "watch": 0, "critical": 0}
        for score in latest.values():
            bands[score.band] = bands.get(score.band, 0) + 1
        for band, count in bands.items():
            self.metrics.devices.set(count, band=band)
        self.metrics.open_alerts.set(len(self.alerts.list(open_only=True)))
        if latest:
            self.metrics.fleet_health_mean.set(sum(s.score for s in latest.values()) / len(latest))
        stages: dict[str, int] = {}
        for version in self.registry.versions():
            stage = self.registry.state(version).stage.value
            stages[stage] = stages.get(stage, 0) + 1
        for stage, count in stages.items():
            self.metrics.releases.set(count, stage=stage)
