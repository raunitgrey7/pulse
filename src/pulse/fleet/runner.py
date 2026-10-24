"""Drive a fleet of simulated devices against the ingest API in rounds."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx
import numpy as np

from pulse.fleet.device import DegradationPlan, DeviceConfig, SimulatedDevice
from pulse.registry.releases import ReleaseManifest
from pulse.registry.signing import VerifyKey
from pulse.signals.generator import Condition, MachineSpec

log = logging.getLogger("pulse.fleet")
RPM_CHOICES = (900.0, 1200.0, 1500.0, 1800.0, 2400.0, 3000.0, 3600.0)


@dataclass(frozen=True)
class FleetConfig:
    n_devices: int = 12
    fault_fraction: float = 0.5
    seed: int = 0
    rounds: int = 60
    interval_s: float = 1.0
    sync_every: int = 10
    fallback_model: Path | None = None
    trust_key_hex: str | None = None


def build_fleet(config: FleetConfig, spec: MachineSpec = MachineSpec()) -> list[SimulatedDevice]:
    """Deterministic device population: ids, speeds, noise and who degrades when."""
    rng = np.random.default_rng(config.seed)
    n_faulty = round(config.n_devices * config.fault_fraction)
    faults = [c for c in Condition if c is not Condition.HEALTHY]
    trust = VerifyKey.from_hex(config.trust_key_hex) if config.trust_key_hex else None
    devices: list[SimulatedDevice] = []
    for i in range(config.n_devices):
        plan = DegradationPlan()
        if i < n_faulty:
            onset = int(rng.integers(config.rounds // 4, max(config.rounds // 4 + 1, config.rounds // 2)))
            plan = DegradationPlan(
                faults[i % len(faults)], onset, max(1, config.rounds // 3), float(rng.uniform(0.6, 1.0))
            )
        device_config = DeviceConfig(
            device_id=f"dev-{i + 1:03d}",
            rpm=float(rng.choice(RPM_CHOICES)),
            snr_db=float(rng.uniform(0.0, 12.0)),
            plan=plan,
            seed=config.seed * 1000 + i,
        )
        devices.append(SimulatedDevice(device_config, spec, trust, config.fallback_model))
    return devices


class RegistryClient:
    """Model source backed by the API, usable with any httpx client (including TestClient)."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def assignment(self, device_id: str) -> str | None:
        response = self._client.get(f"/v1/registry/assignment/{device_id}")
        response.raise_for_status()
        return response.json()["version"]

    def fetch(self, version: str) -> tuple[ReleaseManifest, bytes]:
        meta = self._client.get(f"/v1/registry/releases/{version}")
        meta.raise_for_status()
        model = self._client.get(f"/v1/registry/releases/{version}/model")
        model.raise_for_status()
        return ReleaseManifest.from_dict(meta.json()["manifest"]), model.content


@dataclass
class RunStats:
    rounds: int = 0
    reports_sent: int = 0
    reports_accepted: int = 0
    alerts_opened: int = 0
    model_updates: int = 0
    rejected_releases: int = 0
    post_failures: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


class FleetRunner:
    """Posts one batch per round; syncs models every ``sync_every`` rounds."""

    def __init__(
        self,
        devices: list[SimulatedDevice],
        client: httpx.Client,
        config: FleetConfig,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.devices = devices
        self.client = client
        self.config = config
        self.registry = RegistryClient(client)
        self.clock = clock
        self.sleep = sleep
        self.stats = RunStats()

    def sync_models(self) -> None:
        for device in self.devices:
            try:
                if device.sync_model(self.registry):
                    self.stats.model_updates += 1
            except httpx.HTTPError as exc:
                log.warning("model sync failed", extra={"device_id": device.device_id, "error": str(exc)})
        self.stats.rejected_releases = sum(len(d.stats.rejected_releases) for d in self.devices)

    def round(self) -> None:
        now = self.clock()
        reports = [device.step(now).model_dump() for device in self.devices]
        self.stats.reports_sent += len(reports)
        try:
            response = self.client.post("/v1/telemetry", json={"reports": reports})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            self.stats.post_failures += 1
            log.warning("telemetry post failed", extra={"error": str(exc)})
            return
        body = response.json()
        self.stats.reports_accepted += int(body["accepted"])
        self.stats.alerts_opened += int(body["alerts_opened"])

    def run(self) -> RunStats:
        for i in range(self.config.rounds):
            if i % self.config.sync_every == 0:
                self.sync_models()
            self.round()
            self.stats.rounds += 1
            if self.config.interval_s > 0 and i < self.config.rounds - 1:
                self.sleep(self.config.interval_s)
        return self.stats
