"""Minimal Prometheus text exposition without the client library.

Three metric types are enough for this service and a dependency-free
implementation keeps the container small. Label values are escaped per the
exposition format. ``render`` is deterministic (sorted) so tests can assert
on exact lines.
"""

from __future__ import annotations

import threading
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field

Labels = tuple[tuple[str, str], ...]


def _labels(**kwargs: str) -> Labels:
    return tuple(sorted((k, str(v)) for k, v in kwargs.items()))


def _fmt_labels(labels: Labels) -> str:
    if not labels:
        return ""
    body = ",".join(f'{k}="{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"' for k, v in labels)
    return "{" + body + "}"


@dataclass
class Counter:
    name: str
    help: str
    _values: dict[Labels, float] = field(default_factory=dict)

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        key = _labels(**labels)
        self._values[key] = self._values.get(key, 0.0) + amount

    def value(self, **labels: str) -> float:
        return self._values.get(_labels(**labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        lines += [f"{self.name}{_fmt_labels(k)} {v:g}" for k, v in sorted(self._values.items())]
        return lines


@dataclass
class Gauge:
    name: str
    help: str
    _values: dict[Labels, float] = field(default_factory=dict)

    def set(self, value: float, **labels: str) -> None:
        self._values[_labels(**labels)] = float(value)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        lines += [f"{self.name}{_fmt_labels(k)} {v:g}" for k, v in sorted(self._values.items())]
        return lines


@dataclass
class Histogram:
    name: str
    help: str
    buckets: Sequence[float] = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
    _counts: dict[Labels, list[int]] = field(default_factory=dict)
    _sums: dict[Labels, float] = field(default_factory=dict)
    _totals: dict[Labels, int] = field(default_factory=dict)

    def observe(self, value: float, **labels: str) -> None:
        key = _labels(**labels)
        counts = self._counts.setdefault(key, [0] * len(self.buckets))
        idx = bisect_left(list(self.buckets), value)
        for i in range(idx, len(self.buckets)):
            counts[i] += 1
        self._sums[key] = self._sums.get(key, 0.0) + value
        self._totals[key] = self._totals.get(key, 0) + 1

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, counts in sorted(self._counts.items()):
            for bound, count in zip(self.buckets, counts, strict=True):
                lines.append(f"{self.name}_bucket{_fmt_labels((*key, ('le', f'{bound:g}')))} {count}")
            lines.append(f"{self.name}_bucket{_fmt_labels((*key, ('le', '+Inf')))} {self._totals[key]}")
            lines.append(f"{self.name}_sum{_fmt_labels(key)} {self._sums[key]:g}")
            lines.append(f"{self.name}_count{_fmt_labels(key)} {self._totals[key]}")
        return lines


class MetricsRegistry:
    """The service's metrics, rendered in exposition format under one lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.http_requests = Counter("pulse_http_requests_total", "HTTP requests by method, path template and status.")
        self.http_duration = Histogram("pulse_http_request_duration_seconds", "HTTP request latency.")
        self.telemetry_reports = Counter(
            "pulse_telemetry_reports_total", "Telemetry reports accepted, by predicted class."
        )
        self.alerts_opened = Counter("pulse_alerts_opened_total", "Alerts opened, by severity and rule.")
        self.devices = Gauge("pulse_devices", "Devices known to the fleet, by health band.")
        self.open_alerts = Gauge("pulse_open_alerts", "Alerts currently open.")
        self.fleet_health_mean = Gauge("pulse_fleet_health_mean", "Mean of the latest health score across devices.")
        self.releases = Gauge("pulse_registry_releases", "Releases in the registry, by rollout stage.")

    def render(self) -> str:
        with self._lock:
            metrics = [
                self.http_requests,
                self.http_duration,
                self.telemetry_reports,
                self.alerts_opened,
                self.devices,
                self.open_alerts,
                self.fleet_health_mean,
                self.releases,
            ]
            return "\n".join(line for m in metrics for line in m.render()) + "\n"
