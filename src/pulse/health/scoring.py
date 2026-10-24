"""Health score for one device from its recent telemetry.

Four components, each in [0, 1] where 1 is healthy:

* ``fault``: one minus the exponentially weighted fault probability, so a
  single noisy window cannot flip the score.
* ``level``: vibration RMS relative to the device's own baseline, graded
  like the ISO 10816 severity zones (ratios 1.5, 2.5 and 4 mark the zone
  boundaries A/B, B/C and C/D). Baselines are per device because machines
  differ in mounting and size.
* ``impulsiveness``: kurtosis above 3 (Gaussian) signals impacts, which is
  the earliest sign of a bearing defect, often before the level rises.
* ``trend``: a rising RMS slope over the window costs points even while the
  absolute level is still acceptable.

The weights favour the classifier (it is the most specific evidence) but
keep the physical indicators strong enough to flag a machine the model
misses. Score bands: healthy at or above 70, watch from 40, critical below.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

HEALTHY_MIN = 70.0
WATCH_MIN = 40.0
ZONE_BOUNDS = (1.5, 2.5, 4.0)


@dataclass(frozen=True)
class TelemetryPoint:
    """One ingested window summary as the scorer sees it."""

    timestamp: float
    rms_g: float
    kurtosis: float
    fault_probability: float
    predicted_class: str


@dataclass(frozen=True)
class HealthScore:
    score: float
    band: str
    components: dict[str, float] = field(default_factory=dict)
    zone: str = "A"
    dominant_class: str = "healthy"


def ewma(values: Sequence[float], alpha: float) -> float:
    """Exponentially weighted mean, most recent value last."""
    if not values:
        raise ValueError("ewma of nothing")
    acc = float(values[0])
    for v in values[1:]:
        acc = alpha * float(v) + (1.0 - alpha) * acc
    return acc


def vibration_zone(ratio: float) -> str:
    """ISO 10816 style zone letter for an RMS ratio against baseline."""
    for letter, bound in zip("ABC", ZONE_BOUNDS, strict=True):
        if ratio <= bound:
            return letter
    return "D"


def level_component(ratio: float) -> float:
    """1.0 inside zone A, falling linearly to 0.0 at the C/D boundary."""
    if ratio <= ZONE_BOUNDS[0]:
        return 1.0
    span = ZONE_BOUNDS[2] - ZONE_BOUNDS[0]
    return float(np.clip(1.0 - (ratio - ZONE_BOUNDS[0]) / span, 0.0, 1.0))


def impulsiveness_component(kurtosis: float) -> float:
    """1.0 at Gaussian kurtosis, 0.0 at kurtosis 8 or above."""
    return float(np.clip(1.0 - (kurtosis - 3.0) / 5.0, 0.0, 1.0))


def trend_component(rms_values: Sequence[float], baseline: float) -> float:
    """1.0 for a flat or falling level; 0.0 when RMS rose by a full baseline over the window."""
    if len(rms_values) < 3:
        return 1.0
    x = np.arange(len(rms_values), dtype=np.float64)
    slope = float(np.polyfit(x, np.asarray(rms_values, dtype=np.float64), 1)[0])
    rise = slope * (len(rms_values) - 1) / max(baseline, 1e-9)
    return float(np.clip(1.0 - rise, 0.0, 1.0))


def band_for(score: float) -> str:
    if score >= HEALTHY_MIN:
        return "healthy"
    return "watch" if score >= WATCH_MIN else "critical"


@dataclass(frozen=True)
class HealthScorer:
    """Weights and smoothing for the four components."""

    alpha: float = 0.3
    w_fault: float = 0.5
    w_level: float = 0.2
    w_impulsiveness: float = 0.2
    w_trend: float = 0.1

    def score(self, points: Sequence[TelemetryPoint], baseline_rms: float) -> HealthScore:
        """Score the most recent points (oldest first) against a per-device RMS baseline."""
        if not points:
            raise ValueError("cannot score a device without telemetry")
        baseline = max(baseline_rms, 1e-9)
        rms = [p.rms_g for p in points]
        fault = 1.0 - float(np.clip(ewma([p.fault_probability for p in points], self.alpha), 0.0, 1.0))
        ratio = ewma(rms, self.alpha) / baseline
        level = level_component(ratio)
        impulsive = impulsiveness_component(ewma([p.kurtosis for p in points], self.alpha))
        trend = trend_component(rms, baseline)
        raw = self.w_fault * fault + self.w_level * level + self.w_impulsiveness * impulsive + self.w_trend * trend
        score = float(np.clip(100.0 * raw, 0.0, 100.0))
        classes = [p.predicted_class for p in points[-5:]]
        dominant = max(set(classes), key=classes.count)
        return HealthScore(
            score=round(score, 2),
            band=band_for(score),
            components={"fault": fault, "level": level, "impulsiveness": impulsive, "trend": trend},
            zone=vibration_zone(ratio),
            dominant_class=dominant,
        )
