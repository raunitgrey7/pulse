"""Single-window inference latency.

Edge inference runs one window at a time, so latency is measured at batch
size one, after a warm-up, cycling through different windows so no cache
can hide the cost. Percentiles come from the raw per-call timings.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class LatencyStats:
    """Per-call timings summarised in milliseconds."""

    p50_ms: float
    p95_ms: float
    mean_ms: float
    min_ms: float
    repeats: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def summarise(timings_ns: Sequence[int]) -> LatencyStats:
    """Reduce raw nanosecond timings to the published statistics."""
    if not timings_ns:
        raise ValueError("no timings")
    ms = np.asarray(timings_ns, dtype=np.float64) / 1e6
    return LatencyStats(
        p50_ms=float(np.percentile(ms, 50)),
        p95_ms=float(np.percentile(ms, 95)),
        mean_ms=float(ms.mean()),
        min_ms=float(ms.min()),
        repeats=int(ms.size),
    )


def measure_latency(call: Callable[[int], object], warmup: int, repeats: int) -> LatencyStats:
    """Time ``call(i)`` for i in range(repeats) after ``warmup`` untimed calls."""
    for i in range(warmup):
        call(i)
    timings: list[int] = []
    for i in range(repeats):
        start = time.perf_counter_ns()
        call(i)
        timings.append(time.perf_counter_ns() - start)
    return summarise(timings)
