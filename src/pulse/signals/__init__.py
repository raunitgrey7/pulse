"""Physically motivated synthetic vibration signals for rotating machinery.

The package has three layers. ``bearing`` holds the kinematic formulas that
turn bearing geometry and shaft speed into characteristic fault frequencies.
``generator`` synthesises accelerometer windows for a given machine condition
by superposing shaft harmonics, impact trains that excite a structural
resonance, mains hum and sensor noise. ``dataset`` draws balanced labelled
datasets with varying speed, severity and noise and caches them on disk.
"""

from pulse.signals.bearing import BearingGeometry, FaultFrequencies, fault_frequencies
from pulse.signals.dataset import Dataset, DatasetConfig, make_dataset
from pulse.signals.generator import (
    CLASS_NAMES,
    Condition,
    MachineSpec,
    WindowParams,
    generate_window,
)

__all__ = [
    "CLASS_NAMES",
    "BearingGeometry",
    "Condition",
    "Dataset",
    "DatasetConfig",
    "FaultFrequencies",
    "MachineSpec",
    "WindowParams",
    "fault_frequencies",
    "generate_window",
    "make_dataset",
]
