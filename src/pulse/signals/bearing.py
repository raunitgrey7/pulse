"""Rolling-element bearing kinematics.

A rolling-element bearing has four characteristic frequencies that depend
only on its geometry and the shaft speed. A defect on the outer race is
struck by every rolling element that passes it, a defect on the inner race
passes under every element once per relative revolution, a defect on a ball
strikes both races every half spin, and the cage turns at the mean speed of
the elements. These are the frequencies a vibration analyst looks for in the
envelope spectrum, so the generator and the feature extractor share this
module to stay consistent. The formulas are the standard ones (Randall,
"Vibration-based Condition Monitoring", 2011, section 2.2.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BearingGeometry:
    """Geometry of a deep-groove ball bearing.

    Defaults are the SKF 6205 drive-end bearing used in the Case Western
    Reserve University bearing dataset, so the derived orders can be checked
    against published values (BPFO 3.5848, BPFI 5.4152, BSF 2.3568, FTF 0.3983
    shaft orders).
    """

    n_balls: int = 9
    ball_diameter_mm: float = 7.94
    pitch_diameter_mm: float = 39.04
    contact_angle_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.n_balls < 2:
            raise ValueError("a bearing needs at least two rolling elements")
        if self.ball_diameter_mm <= 0 or self.pitch_diameter_mm <= 0:
            raise ValueError("diameters must be positive")
        if self.ball_diameter_mm >= self.pitch_diameter_mm:
            raise ValueError("ball diameter must be smaller than pitch diameter")
        if not 0.0 <= self.contact_angle_deg < 90.0:
            raise ValueError("contact angle must be in [0, 90) degrees")

    @property
    def diameter_ratio(self) -> float:
        """d/D times the cosine of the contact angle, the term shared by all formulas."""
        return self.ball_diameter_mm / self.pitch_diameter_mm * math.cos(math.radians(self.contact_angle_deg))


@dataclass(frozen=True)
class FaultFrequencies:
    """Characteristic frequencies in hertz for one shaft speed."""

    shaft_hz: float
    bpfo: float
    bpfi: float
    bsf: float
    ftf: float

    def as_orders(self) -> FaultFrequencies:
        """Return the same frequencies divided by the shaft frequency."""
        f = self.shaft_hz
        return FaultFrequencies(1.0, self.bpfo / f, self.bpfi / f, self.bsf / f, self.ftf / f)


def fault_frequencies(geometry: BearingGeometry, shaft_hz: float) -> FaultFrequencies:
    """Compute BPFO, BPFI, BSF and FTF for a shaft speed in hertz.

    BPFO = n/2 * fr * (1 - r), BPFI = n/2 * fr * (1 + r), BSF = D/(2d) * fr * (1 - r^2)
    and FTF = fr/2 * (1 - r) with r = d/D * cos(alpha). BPFO + BPFI equals n * fr,
    which the tests use as an identity.
    """
    if shaft_hz <= 0:
        raise ValueError("shaft frequency must be positive")
    r = geometry.diameter_ratio
    n = geometry.n_balls
    bpfo = n / 2.0 * shaft_hz * (1.0 - r)
    bpfi = n / 2.0 * shaft_hz * (1.0 + r)
    bsf = geometry.pitch_diameter_mm / (2.0 * geometry.ball_diameter_mm) * shaft_hz * (1.0 - r * r)
    ftf = shaft_hz / 2.0 * (1.0 - r)
    return FaultFrequencies(shaft_hz, bpfo, bpfi, bsf, ftf)
