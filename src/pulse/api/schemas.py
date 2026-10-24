"""Typed request and response bodies.

Validation is strict on purpose: device ids are constrained to a safe
alphabet, every float must be finite and inside a physically plausible
range, probability vectors must have one entry per class and sum to one
within tolerance, and batches are capped. Bad telemetry is rejected with a
422 before it can reach the store or the scorer.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pulse.signals.generator import CLASS_NAMES

DeviceId = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]
Finite = Annotated[float, Field(allow_inf_nan=False)]
ClassName = Literal["healthy", "imbalance", "misalignment", "outer_race", "inner_race", "ball"]
Version = Annotated[str, Field(pattern=r"^\d+\.\d+\.\d+$")]


class TelemetryReport(BaseModel):
    """One scored window as reported by a device."""

    model_config = ConfigDict(extra="forbid")

    device_id: DeviceId
    timestamp: Annotated[float, Field(gt=0, allow_inf_nan=False)]
    rpm: Annotated[float, Field(gt=0, lt=100_000, allow_inf_nan=False)]
    rms_g: Annotated[float, Field(ge=0, lt=1_000, allow_inf_nan=False)]
    kurtosis: Annotated[float, Field(ge=1, lt=10_000, allow_inf_nan=False)]
    crest_factor: Annotated[float, Field(ge=1, lt=10_000, allow_inf_nan=False)]
    predicted_class: ClassName
    probabilities: Annotated[list[Finite], Field(min_length=len(CLASS_NAMES), max_length=len(CLASS_NAMES))]
    model_version: Annotated[str, Field(min_length=1, max_length=64)]
    latency_ms: Annotated[float, Field(ge=0, lt=600_000, allow_inf_nan=False)]
    status: Literal["ok", "error"] = "ok"
    temperature_c: Annotated[float, Field(gt=-100, lt=400, allow_inf_nan=False)] | None = None

    @field_validator("probabilities")
    @classmethod
    def _probabilities_are_a_distribution(cls, values: list[float]) -> list[float]:
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError("probabilities must lie in [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-3):
            raise ValueError("probabilities must sum to 1")
        return values

    @property
    def fault_probability(self) -> float:
        return 1.0 - self.probabilities[CLASS_NAMES.index("healthy")]


class TelemetryBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reports: Annotated[list[TelemetryReport], Field(min_length=1, max_length=500)]


class IngestResponse(BaseModel):
    accepted: int
    devices_updated: int
    alerts_opened: int


class HealthScoreOut(BaseModel):
    score: float
    band: str
    zone: str
    dominant_class: str
    components: dict[str, float]


class DeviceSummary(BaseModel):
    device_id: str
    first_seen: float
    last_seen: float
    model_version: str
    n_reports: int
    baseline_rms_g: float | None
    health: HealthScoreOut | None
    recent_scores: list[float]
    last_rpm: float | None
    last_rms_g: float | None


class TelemetryPointOut(BaseModel):
    timestamp: float
    rpm: float
    rms_g: float
    kurtosis: float
    crest_factor: float
    predicted_class: str
    fault_probability: float
    model_version: str
    latency_ms: float
    status: str


class DeviceDetail(DeviceSummary):
    recent: list[TelemetryPointOut]


class AlertOut(BaseModel):
    id: int
    device_id: str
    rule: str
    severity: str
    message: str
    created_at: float
    updated_at: float
    occurrences: int
    acknowledged: bool
    resolved_at: float | None
    open: bool
    details: dict[str, float | str]


class FleetSummary(BaseModel):
    devices: int
    bands: dict[str, int]
    mean_score: float | None
    open_alerts: int
    versions: dict[str, int]
    reports_total: int


class ReleaseIn(BaseModel):
    """A signed manifest plus the model bytes it describes."""

    model_config = ConfigDict(extra="forbid")

    manifest: dict[str, object]
    model_b64: Annotated[str, Field(min_length=1, max_length=64_000_000)]
    signer_public_key_hex: Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


class RolloutOut(BaseModel):
    version: str
    stage: str
    percent: int
    history: list[dict[str, str]]


class ReleaseOut(BaseModel):
    manifest: dict[str, object]
    rollout: RolloutOut
    cohort: dict[str, float | int | str] | None = None


class RollbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: Annotated[str, Field(min_length=1, max_length=500)]


class AssignmentOut(BaseModel):
    device_id: str
    bucket: int
    version: str | None
    manifest: dict[str, object] | None
    model_path: str | None


class ServiceHealth(BaseModel):
    status: Literal["ok"]
    version: str
    uptime_s: float
    devices: int
    open_alerts: int
    releases: int
