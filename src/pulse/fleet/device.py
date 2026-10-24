"""One simulated edge device: generate a window, score it locally, report a summary.

The device only ever sends a summary (RMS, kurtosis, crest factor, class
probabilities), never the raw window, because that is the bandwidth budget
of a real cellular or LoRa deployment. Model updates follow the registry's
assignment and are verified against the device's trust key; a tampered or
unsigned release is rejected and counted, and the device keeps running the
model it has.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from pulse.api.schemas import TelemetryReport
from pulse.models.runtime import OnnxPredictor
from pulse.registry.releases import ReleaseManifest, verify_release
from pulse.registry.signing import VerifyKey
from pulse.signals.generator import CLASS_NAMES, Condition, MachineSpec, WindowParams, generate_window

log = logging.getLogger("pulse.device")


class ModelSource(Protocol):
    """Where a device asks for its model: the registry API or an in-process stand-in."""

    def assignment(self, device_id: str) -> str | None: ...

    def fetch(self, version: str) -> tuple[ReleaseManifest, bytes]: ...


@dataclass(frozen=True)
class DegradationPlan:
    """Healthy until ``onset_step``, then a fault whose severity ramps up linearly."""

    fault: Condition | None = None
    onset_step: int = 0
    ramp_steps: int = 1
    final_severity: float = 1.0
    initial_severity: float = 0.05

    def at(self, step: int) -> tuple[Condition, float]:
        if self.fault is None or step < self.onset_step:
            return Condition.HEALTHY, 0.0
        progress = min(1.0, (step - self.onset_step) / max(1, self.ramp_steps))
        return self.fault, self.initial_severity + (self.final_severity - self.initial_severity) * progress


@dataclass(frozen=True)
class DeviceConfig:
    device_id: str
    rpm: float
    snr_db: float
    plan: DegradationPlan
    seed: int
    speed_jitter: float = 0.01


@dataclass
class DeviceStats:
    reports: int = 0
    inference_errors: int = 0
    model_updates: int = 0
    rejected_releases: list[str] = field(default_factory=list)


class SimulatedDevice:
    """Runs the edge loop: sense, infer, summarise, report, occasionally sync the model."""

    def __init__(
        self,
        config: DeviceConfig,
        spec: MachineSpec,
        trust_key: VerifyKey | None,
        fallback_model: Path | None,
        fallback_version: str = "builtin",
    ) -> None:
        self.config = config
        self.spec = spec
        self.trust_key = trust_key
        self.rng = np.random.default_rng(config.seed)
        self.predictor: OnnxPredictor | None = OnnxPredictor(fallback_model) if fallback_model else None
        self.model_version = fallback_version
        self.stats = DeviceStats()
        self.step_index = 0

    @property
    def device_id(self) -> str:
        return self.config.device_id

    def sync_model(self, source: ModelSource) -> bool:
        """Adopt the assigned release if it verifies; return True when the model changed."""
        version = source.assignment(self.device_id)
        if version is None or version == self.model_version or version in self.stats.rejected_releases:
            return False
        manifest, model_bytes = source.fetch(version)
        if self.trust_key is None:
            self.stats.rejected_releases.append(version)
            log.warning(
                "no trust key configured, refusing release", extra={"device_id": self.device_id, "version": version}
            )
            return False
        result = verify_release(self.trust_key, manifest, model_bytes)
        if not result.ok:
            self.stats.rejected_releases.append(version)
            log.warning(
                "release rejected", extra={"device_id": self.device_id, "version": version, "reason": result.reason}
            )
            return False
        self.predictor = OnnxPredictor(model_bytes, name=f"onnx-{version}")
        self.model_version = version
        self.stats.model_updates += 1
        return True

    def _sense(self) -> tuple[np.ndarray, float]:
        condition, severity = self.config.plan.at(self.step_index)
        rpm = self.config.rpm * (1.0 + self.config.speed_jitter * float(self.rng.standard_normal()))
        params = WindowParams(condition, rpm, severity, self.config.snr_db)
        return generate_window(self.spec, params, self.rng).signal, rpm

    def _infer(self, window: np.ndarray, rpm: float) -> tuple[np.ndarray, float, str]:
        if self.predictor is None:
            return np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float32), 0.0, "error"
        start = time.perf_counter()
        try:
            probs = self.predictor.predict_proba(window[None, :], np.asarray([rpm]))[0]
        except (RuntimeError, ValueError) as exc:
            self.stats.inference_errors += 1
            log.error("inference failed", extra={"device_id": self.device_id, "error": str(exc)})
            return np.full(len(CLASS_NAMES), 1.0 / len(CLASS_NAMES), dtype=np.float32), 0.0, "error"
        return probs, (time.perf_counter() - start) * 1000.0, "ok"

    def step(self, timestamp: float) -> TelemetryReport:
        """Produce one report for the current step and advance the degradation plan."""
        window, rpm = self._sense()
        probs, latency_ms, status = self._infer(window, rpm)
        centred = window.astype(np.float64) - window.mean()
        rms = float(np.sqrt(np.mean(centred**2))) + 1e-12
        kurtosis = float(np.mean(centred**4) / (np.mean(centred**2) ** 2 + 1e-24))
        crest = float(np.abs(centred).max() / rms)
        probs = probs / probs.sum()
        self.step_index += 1
        self.stats.reports += 1
        return TelemetryReport(
            device_id=self.device_id,
            timestamp=timestamp,
            rpm=rpm,
            rms_g=rms,
            kurtosis=max(1.0, kurtosis),
            crest_factor=max(1.0, crest),
            predicted_class=CLASS_NAMES[int(probs.argmax())],
            probabilities=[float(p) for p in probs],
            model_version=self.model_version,
            latency_ms=latency_ms,
            status=status,  # type: ignore[arg-type]
        )
