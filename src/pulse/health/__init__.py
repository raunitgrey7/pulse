"""Device health: a 0 to 100 score from recent telemetry, and rules that raise alerts.

Classifier output alone is a poor operator signal: it flickers window to
window and says nothing about trend. ``scoring`` combines a smoothed fault
probability with vibration level relative to the device's own baseline,
impulsiveness and the level trend into one score with named components.
``alerts`` turns scores and predictions into deduplicated alerts with
severities, acknowledgement and automatic resolution.
"""

from pulse.health.alerts import Alert, AlertManager, AlertRules, Severity
from pulse.health.scoring import HealthScore, HealthScorer, TelemetryPoint

__all__ = ["Alert", "AlertManager", "AlertRules", "HealthScore", "HealthScorer", "Severity", "TelemetryPoint"]
