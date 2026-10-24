"""Alert rules, deduplication and lifecycle.

Rules are evaluated per device on every ingest. Each (device, rule) pair has
at most one open alert, so a persistent condition produces one alert with
a growing occurrence count rather than a flood. When the condition clears
the alert resolves itself; operators can also acknowledge it, which keeps
it visible but stops it counting as new. Sinks receive every newly opened
alert; the default sink logs, an optional webhook sink posts JSON.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum

from pulse.health.scoring import HealthScore

log = logging.getLogger("pulse.alerts")


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    id: int
    device_id: str
    rule: str
    severity: Severity
    message: str
    created_at: float
    updated_at: float
    occurrences: int = 1
    acknowledged: bool = False
    resolved_at: float | None = None
    details: dict[str, float | str] = field(default_factory=dict)

    @property
    def open(self) -> bool:
        return self.resolved_at is None

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "severity": self.severity.value, "open": self.open}


@dataclass(frozen=True)
class Candidate:
    """A rule's verdict before deduplication."""

    rule: str
    severity: Severity
    message: str
    details: dict[str, float | str]


@dataclass(frozen=True)
class AlertRules:
    """Thresholds for the built-in rules."""

    critical_score: float = 40.0
    critical_consecutive: int = 3
    fault_probability: float = 0.8
    fault_consecutive: int = 5
    stale_after_s: float = 60.0

    def evaluate(
        self,
        scores: Sequence[HealthScore],
        fault_probabilities: Sequence[float],
        predicted_classes: Sequence[str],
    ) -> list[Candidate]:
        """Rules over the most recent history (oldest first)."""
        found: list[Candidate] = []
        recent = scores[-self.critical_consecutive :]
        if len(recent) == self.critical_consecutive and all(s.score < self.critical_score for s in recent):
            found.append(
                Candidate(
                    "health_critical",
                    Severity.CRITICAL,
                    f"health {recent[-1].score:.0f} below {self.critical_score:.0f} for {len(recent)} windows",
                    {"score": recent[-1].score, "dominant_class": recent[-1].dominant_class},
                )
            )
        last_p = fault_probabilities[-self.fault_consecutive :]
        last_c = predicted_classes[-self.fault_consecutive :]
        if (
            len(last_p) == self.fault_consecutive
            and all(p >= self.fault_probability for p in last_p)
            and len(set(last_c)) == 1
            and last_c[0] != "healthy"
        ):
            found.append(
                Candidate(
                    "fault_persistent",
                    Severity.WARNING,
                    f"{last_c[0]} predicted with p >= {self.fault_probability} for {len(last_p)} windows",
                    {"predicted_class": last_c[0], "probability": float(last_p[-1])},
                )
            )
        if scores and scores[-1].zone == "D":
            found.append(
                Candidate(
                    "vibration_zone_d",
                    Severity.CRITICAL,
                    "vibration level in zone D (more than 4x baseline)",
                    {"zone": "D"},
                )
            )
        return found


Sink = Callable[[Alert], None]


def log_sink(alert: Alert) -> None:
    log.warning("alert opened", extra={"alert": alert.to_dict()})


class AlertManager:
    """Open, update, resolve and acknowledge alerts; one open alert per device and rule."""

    def __init__(self, sinks: Sequence[Sink] | None = None, clock: Callable[[], float] = time.time) -> None:
        self._alerts: dict[int, Alert] = {}
        self._open: dict[tuple[str, str], int] = {}
        self._next_id = 1
        self._sinks = list(sinks) if sinks is not None else [log_sink]
        self._clock = clock

    def apply(self, device_id: str, candidates: Sequence[Candidate]) -> list[Alert]:
        """Reconcile a device's current candidates with its open alerts; return newly opened ones."""
        now = self._clock()
        opened: list[Alert] = []
        active = {c.rule for c in candidates}
        for candidate in candidates:
            key = (device_id, candidate.rule)
            if key in self._open:
                alert = self._alerts[self._open[key]]
                alert.occurrences += 1
                alert.updated_at = now
                alert.details = dict(candidate.details)
                continue
            alert = Alert(
                self._next_id,
                device_id,
                candidate.rule,
                candidate.severity,
                candidate.message,
                now,
                now,
                details=dict(candidate.details),
            )
            self._next_id += 1
            self._alerts[alert.id] = alert
            self._open[key] = alert.id
            opened.append(alert)
        for key in [k for k in self._open if k[0] == device_id and k[1] not in active]:
            self._alerts[self._open.pop(key)].resolved_at = now
        for alert in opened:
            for sink in self._sinks:
                sink(alert)
        return opened

    def stale(self, device_id: str, last_seen: float, stale_after_s: float) -> Alert | None:
        """Open (or refresh) a staleness alert when a device has gone quiet."""
        if self._clock() - last_seen < stale_after_s:
            self.apply_rule_clear(device_id, "device_stale")
            return None
        candidate = Candidate(
            "device_stale", Severity.WARNING, f"no telemetry for {stale_after_s:.0f} s", {"last_seen": last_seen}
        )
        return self.apply_single(device_id, candidate)

    def apply_single(self, device_id: str, candidate: Candidate) -> Alert | None:
        """Apply one candidate without resolving the device's other open alerts."""
        key = (device_id, candidate.rule)
        now = self._clock()
        if key in self._open:
            alert = self._alerts[self._open[key]]
            alert.occurrences += 1
            alert.updated_at = now
            return None
        alert = Alert(
            self._next_id,
            device_id,
            candidate.rule,
            candidate.severity,
            candidate.message,
            now,
            now,
            details=dict(candidate.details),
        )
        self._next_id += 1
        self._alerts[alert.id] = alert
        self._open[key] = alert.id
        for sink in self._sinks:
            sink(alert)
        return alert

    def apply_rule_clear(self, device_id: str, rule: str) -> None:
        key = (device_id, rule)
        if key in self._open:
            self._alerts[self._open.pop(key)].resolved_at = self._clock()

    def acknowledge(self, alert_id: int) -> Alert:
        alert = self._alerts[alert_id]
        alert.acknowledged = True
        alert.updated_at = self._clock()
        return alert

    def get(self, alert_id: int) -> Alert | None:
        return self._alerts.get(alert_id)

    def list(self, open_only: bool = False, device_id: str | None = None) -> list[Alert]:
        alerts = [
            a
            for a in self._alerts.values()
            if (not open_only or a.open) and (device_id is None or a.device_id == device_id)
        ]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
