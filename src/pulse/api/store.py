"""SQLite persistence for devices, telemetry and health scores.

SQLite is enough for a fleet of hundreds of devices reporting every second,
needs no extra service, and keeps the deployment to one container and one
volume. Writes go through a single connection guarded by a lock; reads use
indexed queries on (device_id, timestamp). A device's RMS baseline is fixed
from the median of its first reports, on the assumption that machines are
commissioned in a healthy state, and never moves afterwards so a slow
degradation cannot hide by dragging its own reference along.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from pulse.api.schemas import TelemetryReport
from pulse.health.scoring import HealthScore, TelemetryPoint
from pulse.registry.rollout import CohortStats

BASELINE_REPORTS = 20
SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    model_version TEXT NOT NULL,
    n_reports INTEGER NOT NULL DEFAULT 0,
    baseline_rms REAL
);
CREATE TABLE IF NOT EXISTS telemetry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    rpm REAL NOT NULL,
    rms_g REAL NOT NULL,
    kurtosis REAL NOT NULL,
    crest_factor REAL NOT NULL,
    predicted_class TEXT NOT NULL,
    fault_probability REAL NOT NULL,
    model_version TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS telemetry_device_ts ON telemetry (device_id, timestamp);
CREATE INDEX IF NOT EXISTS telemetry_version ON telemetry (model_version);
CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    timestamp REAL NOT NULL,
    score REAL NOT NULL,
    band TEXT NOT NULL,
    zone TEXT NOT NULL,
    dominant_class TEXT NOT NULL,
    c_fault REAL NOT NULL,
    c_level REAL NOT NULL,
    c_impulsiveness REAL NOT NULL,
    c_trend REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS scores_device_ts ON scores (device_id, timestamp);
"""


@dataclass(frozen=True)
class DeviceRow:
    device_id: str
    first_seen: float
    last_seen: float
    model_version: str
    n_reports: int
    baseline_rms: float | None


def _score_from_row(row: sqlite3.Row) -> HealthScore:
    return HealthScore(
        score=row["score"],
        band=row["band"],
        zone=row["zone"],
        dominant_class=row["dominant_class"],
        components={
            "fault": row["c_fault"],
            "level": row["c_level"],
            "impulsiveness": row["c_impulsiveness"],
            "trend": row["c_trend"],
        },
    )


class TelemetryStore:
    """All reads and writes for the fleet state."""

    def __init__(self, path: Path | str) -> None:
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def close(self) -> None:
        self._conn.close()

    def insert_report(self, report: TelemetryReport) -> DeviceRow:
        """Store one report, update the device row and fix its baseline when due."""
        with self._lock:
            self._conn.execute("BEGIN")
            self._conn.execute(
                "INSERT INTO telemetry (device_id, timestamp, rpm, rms_g, kurtosis, crest_factor, predicted_class,"
                " fault_probability, model_version, latency_ms, status) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    report.device_id,
                    report.timestamp,
                    report.rpm,
                    report.rms_g,
                    report.kurtosis,
                    report.crest_factor,
                    report.predicted_class,
                    report.fault_probability,
                    report.model_version,
                    report.latency_ms,
                    report.status,
                ),
            )
            self._conn.execute(
                "INSERT INTO devices (device_id, first_seen, last_seen, model_version, n_reports) VALUES (?,?,?,?,1)"
                " ON CONFLICT(device_id) DO UPDATE SET last_seen=excluded.last_seen,"
                " model_version=excluded.model_version, n_reports=n_reports+1",
                (report.device_id, report.timestamp, report.timestamp, report.model_version),
            )
            row = self._device_row(report.device_id)
            if row.baseline_rms is None and row.n_reports >= BASELINE_REPORTS:
                row = self._fix_baseline(report.device_id)
            self._conn.execute("COMMIT")
        return row

    def _fix_baseline(self, device_id: str) -> DeviceRow:
        rows = self._conn.execute(
            "SELECT rms_g FROM telemetry WHERE device_id=? ORDER BY timestamp ASC, id ASC LIMIT ?",
            (device_id, BASELINE_REPORTS),
        ).fetchall()
        baseline = median(r["rms_g"] for r in rows)
        self._conn.execute("UPDATE devices SET baseline_rms=? WHERE device_id=?", (baseline, device_id))
        return self._device_row(device_id)

    def _device_row(self, device_id: str) -> DeviceRow:
        row = self._conn.execute("SELECT * FROM devices WHERE device_id=?", (device_id,)).fetchone()
        return DeviceRow(
            row["device_id"],
            row["first_seen"],
            row["last_seen"],
            row["model_version"],
            row["n_reports"],
            row["baseline_rms"],
        )

    def device(self, device_id: str) -> DeviceRow | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM devices WHERE device_id=?", (device_id,)).fetchone()
        if row is None:
            return None
        return DeviceRow(
            row["device_id"],
            row["first_seen"],
            row["last_seen"],
            row["model_version"],
            row["n_reports"],
            row["baseline_rms"],
        )

    def devices(self) -> list[DeviceRow]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM devices ORDER BY device_id").fetchall()
        return [
            DeviceRow(
                r["device_id"], r["first_seen"], r["last_seen"], r["model_version"], r["n_reports"], r["baseline_rms"]
            )
            for r in rows
        ]

    def recent_points(self, device_id: str, limit: int) -> list[TelemetryPoint]:
        """Most recent telemetry, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT timestamp, rms_g, kurtosis, fault_probability, predicted_class FROM telemetry"
                " WHERE device_id=? ORDER BY timestamp DESC, id DESC LIMIT ?",
                (device_id, limit),
            ).fetchall()
        return [
            TelemetryPoint(r["timestamp"], r["rms_g"], r["kurtosis"], r["fault_probability"], r["predicted_class"])
            for r in reversed(rows)
        ]

    def recent_rows(self, device_id: str, limit: int) -> list[dict[str, object]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM telemetry WHERE device_id=? ORDER BY timestamp DESC, id DESC LIMIT ?", (device_id, limit)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def early_baseline(self, device_id: str) -> float:
        """Provisional baseline before BASELINE_REPORTS reports exist: median so far."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT rms_g FROM telemetry WHERE device_id=? ORDER BY timestamp ASC, id ASC LIMIT ?",
                (device_id, BASELINE_REPORTS),
            ).fetchall()
        return median(r["rms_g"] for r in rows) if rows else 1.0

    def insert_score(self, device_id: str, timestamp: float, score: HealthScore) -> None:
        c = score.components
        with self._lock:
            self._conn.execute(
                "INSERT INTO scores (device_id, timestamp, score, band, zone, dominant_class, c_fault, c_level,"
                " c_impulsiveness, c_trend) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    device_id,
                    timestamp,
                    score.score,
                    score.band,
                    score.zone,
                    score.dominant_class,
                    c["fault"],
                    c["level"],
                    c["impulsiveness"],
                    c["trend"],
                ),
            )

    def recent_scores(self, device_id: str, limit: int) -> list[HealthScore]:
        """Most recent scores, oldest first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM scores WHERE device_id=? ORDER BY timestamp DESC, id DESC LIMIT ?", (device_id, limit)
            ).fetchall()
        return [_score_from_row(r) for r in reversed(rows)]

    def latest_scores(self) -> dict[str, HealthScore]:
        """The newest score of every device in one query."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.* FROM scores s JOIN (SELECT device_id, MAX(id) AS mid FROM scores GROUP BY device_id) m"
                " ON s.id = m.mid"
            ).fetchall()
        return {r["device_id"]: _score_from_row(r) for r in rows}

    def cohort_stats(self, version: str) -> CohortStats:
        """Evidence about one release: devices, reports, fault rate and error rate."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT device_id) AS n_devices, COUNT(*) AS n_reports,"
                " AVG(CASE WHEN predicted_class != 'healthy' THEN 1.0 ELSE 0.0 END) AS fault_rate,"
                " AVG(CASE WHEN status != 'ok' THEN 1.0 ELSE 0.0 END) AS error_rate"
                " FROM telemetry WHERE model_version=?",
                (version,),
            ).fetchone()
        return CohortStats(
            version,
            row["n_devices"],
            row["n_reports"],
            float(row["fault_rate"] or 0.0),
            float(row["error_rate"] or 0.0),
        )

    def version_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT model_version, COUNT(*) AS n FROM devices GROUP BY model_version"
            ).fetchall()
        return {r["model_version"]: r["n"] for r in rows}

    def report_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM telemetry").fetchone()[0])

    def device_ids(self) -> Iterable[str]:
        return [d.device_id for d in self.devices()]
