"""Device simulator fleet.

Each simulated device owns a machine (speed, noise level, a degradation
plan), runs the deployed ONNX model locally on every window, summarises the
window and reports the result. Devices pull their model assignment from the
registry, verify the signature and hash before loading, and keep their
current model when verification fails. The fleet runner drives many devices
against the ingest API in lock-step rounds so a demo or a test is
deterministic for a given seed.
"""

from pulse.fleet.device import DegradationPlan, DeviceConfig, SimulatedDevice
from pulse.fleet.runner import FleetConfig, FleetRunner, RegistryClient, build_fleet

__all__ = [
    "DegradationPlan",
    "DeviceConfig",
    "FleetConfig",
    "FleetRunner",
    "RegistryClient",
    "SimulatedDevice",
    "build_fleet",
]
