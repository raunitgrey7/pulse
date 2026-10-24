"""Staged rollout: draft, canary, pilot, fleet, with a health guard.

Devices are assigned to a release by a deterministic bucket (a hash of the
device id into 0..99), so the same devices form the canary cohort for every
release and a device's assignment never flips between polls. A release
moves up the ladder only through ``promote``; moving past canary requires
the guard to pass, which compares the candidate cohort's fault rate and
error rate against the incumbent release. ``rollback`` drops a release to
zero percent immediately. State lives on disk as plain JSON next to the
model files so it can be inspected and mounted into a container.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pulse.registry.releases import ReleaseManifest, parse_version


class Stage(StrEnum):
    DRAFT = "draft"
    CANARY = "canary"
    PILOT = "pilot"
    FLEET = "fleet"
    ROLLED_BACK = "rolled_back"


STAGE_PERCENT: dict[Stage, int] = {
    Stage.DRAFT: 0,
    Stage.CANARY: 5,
    Stage.PILOT: 25,
    Stage.FLEET: 100,
    Stage.ROLLED_BACK: 0,
}
LADDER: tuple[Stage, ...] = (Stage.DRAFT, Stage.CANARY, Stage.PILOT, Stage.FLEET)


def device_bucket(device_id: str, salt: str = "pulse") -> int:
    """Stable bucket in 0..99 for a device id."""
    digest = hashlib.sha256(f"{salt}:{device_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") % 100


@dataclass(frozen=True)
class CohortStats:
    """What the telemetry store knows about the devices running one release."""

    version: str
    n_devices: int
    n_reports: int
    fault_rate: float
    error_rate: float


@dataclass(frozen=True)
class GuardDecision:
    ok: bool
    reason: str


@dataclass(frozen=True)
class CanaryGuard:
    """Promotion policy: enough evidence, few errors, no alarm storm versus the incumbent."""

    min_reports: int = 50
    max_error_rate: float = 0.05
    max_fault_rate_ratio: float = 2.0
    min_fault_rate_floor: float = 0.05

    def decide(self, candidate: CohortStats, incumbent: CohortStats | None) -> GuardDecision:
        if candidate.n_reports < self.min_reports:
            return GuardDecision(False, f"only {candidate.n_reports} reports, need {self.min_reports}")
        if candidate.error_rate > self.max_error_rate:
            return GuardDecision(False, f"error rate {candidate.error_rate:.3f} above {self.max_error_rate}")
        if incumbent is not None and incumbent.n_reports >= self.min_reports:
            floor = max(incumbent.fault_rate, self.min_fault_rate_floor)
            if candidate.fault_rate > self.max_fault_rate_ratio * floor:
                return GuardDecision(
                    False,
                    f"fault rate {candidate.fault_rate:.3f} is more than {self.max_fault_rate_ratio}x "
                    f"the incumbent's {incumbent.fault_rate:.3f}",
                )
        return GuardDecision(True, "canary cohort healthy")


@dataclass
class RolloutState:
    """Where a release stands and how it got there."""

    version: str
    stage: Stage = Stage.DRAFT
    history: list[dict[str, str]] = field(default_factory=list)

    @property
    def percent(self) -> int:
        return STAGE_PERCENT[self.stage]

    def record(self, stage: Stage, reason: str) -> None:
        self.stage = stage
        self.history.append(
            {"stage": stage.value, "at": datetime.now(UTC).isoformat(timespec="seconds"), "reason": reason}
        )


class ModelRegistry:
    """Releases on disk, each in ``<root>/<version>/`` with model, manifest and rollout state."""

    def __init__(self, root: Path, salt: str = "pulse") -> None:
        self.root = root
        self.salt = salt
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, version: str) -> Path:
        parse_version(version)
        return self.root / version

    def versions(self) -> list[str]:
        """All releases, newest first."""
        found = [p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").exists()]
        return sorted(found, key=parse_version, reverse=True)

    def publish(self, manifest: ReleaseManifest, model_bytes: bytes) -> RolloutState:
        target = self._dir(manifest.version)
        if target.exists():
            raise FileExistsError(f"release {manifest.version} already exists; versions are immutable")
        target.mkdir(parents=True)
        (target / "model.bin").write_bytes(model_bytes)
        (target / "manifest.json").write_text(manifest.to_json())
        state = RolloutState(manifest.version)
        state.record(Stage.DRAFT, "published")
        self._save_state(state)
        return state

    def manifest(self, version: str) -> ReleaseManifest:
        return ReleaseManifest.from_json((self._dir(version) / "manifest.json").read_text())

    def model_bytes(self, version: str) -> bytes:
        return (self._dir(version) / "model.bin").read_bytes()

    def state(self, version: str) -> RolloutState:
        raw = json.loads((self._dir(version) / "rollout.json").read_text())
        return RolloutState(raw["version"], Stage(raw["stage"]), raw["history"])

    def _save_state(self, state: RolloutState) -> None:
        payload = {**asdict(state), "stage": state.stage.value}
        (self._dir(state.version) / "rollout.json").write_text(json.dumps(payload, indent=2))

    def promote(
        self,
        version: str,
        guard: CanaryGuard | None = None,
        candidate: CohortStats | None = None,
        incumbent: CohortStats | None = None,
    ) -> RolloutState:
        """Move one step up the ladder; past canary the guard must approve."""
        state = self.state(version)
        if state.stage not in LADDER or state.stage is Stage.FLEET:
            raise ValueError(f"release {version} is at {state.stage.value} and cannot be promoted")
        next_stage = LADDER[LADDER.index(state.stage) + 1]
        reason = "promoted"
        if state.stage is not Stage.DRAFT:
            if guard is None or candidate is None:
                raise ValueError("promotion past canary needs a guard and cohort statistics")
            decision = guard.decide(candidate, incumbent)
            if not decision.ok:
                raise PermissionError(f"promotion blocked: {decision.reason}")
            reason = decision.reason
        state.record(next_stage, reason)
        self._save_state(state)
        return state

    def rollback(self, version: str, reason: str) -> RolloutState:
        state = self.state(version)
        state.record(Stage.ROLLED_BACK, reason)
        self._save_state(state)
        return state

    def incumbent(self, exclude: str | None = None) -> str | None:
        """The newest release at fleet stage, which is what most devices run."""
        for version in self.versions():
            if version != exclude and self.state(version).stage is Stage.FLEET:
                return version
        return None

    def assignment(self, device_id: str) -> str | None:
        """Newest release whose rollout percentage covers this device's bucket."""
        bucket = device_bucket(device_id, self.salt)
        for version in self.versions():
            if bucket < self.state(version).percent:
                return version
        return None
