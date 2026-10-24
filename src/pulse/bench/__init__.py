"""Benchmark harness: trains every variant, measures it and gates on thresholds.

Every number the README quotes comes from ``python -m pulse.bench``. The
harness draws the data with fixed seeds, trains the baseline and the CNN,
exports and quantises, then measures accuracy, artefact size, single-window
latency and the peak memory of a fresh process for each variant on the same
test set. ``check`` turns the results into a pass or fail against explicit
thresholds so CI can use the benchmark as a quality gate.
"""
