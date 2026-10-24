# Evaluation

## What is measured

Every variant is measured on the same held-out test set with the same seeds
by `uv run python -m pulse.bench`, which writes `benchmarks/results.json`
(source of truth) and `benchmarks/RESULTS.md` (the table copied into the
README). Nothing in the README is typed by hand.

| Metric | Definition | How |
|---|---|---|
| Accuracy | Fraction of test windows whose arg-max class equals the label | `sklearn` on the full test set |
| Macro F1 | Unweighted mean of per-class F1 | `f1_score(average="macro")` |
| Per-class recall and confusion matrix | Diagnostic detail per variant | stored in `results.json` |
| Size | Bytes of the artefact as loaded by a device (pickle, state dict or ONNX file) | `Path.stat()` |
| Latency p50, p95, mean, min | Wall time of one single-window prediction, batch size one, after 30 warm-up calls, over 300 different windows, one intra-op thread | `time.perf_counter_ns` around `predict_proba` |
| Peak RSS | Peak resident memory of a fresh interpreter that imports only what the variant needs, loads it and scores every test window one at a time, with `OMP_NUM_THREADS=1` and friends pinned | `pulse.bench.memprobe` via `resource.getrusage` |
| Load time | Import plus artefact load inside that fresh interpreter | `time.perf_counter` |
| Parity | Max absolute probability difference and arg-max agreement between each ONNX variant and the PyTorch float model | computed in `bench.run` |

Latency is single-threaded on purpose: an edge gateway or industrial PC
gives a model one core, and multi-threaded numbers on a laptop would say
nothing about that. The baseline's latency includes feature extraction
(FFT, band-pass filtering, Hilbert transform), because that is what a
device would have to run.

Peak RSS is measured in a fresh process rather than inside the benchmark
process, because memory is a property of the whole runtime: an int8 graph
is 90 KB, but what a device pays for is onnxruntime (about 70 MB here)
versus PyTorch (about 200 MB). Measuring in-process would charge every
variant for every library the harness itself imported.

## Dataset

Synthetic, from the physical model in `pulse.signals` (see
`docs/ARCHITECTURE.md` and the module docstring of `generator.py`). The
generator is documented and tested against the physics it claims: the
envelope spectrum of an outer-race window peaks at BPFO, impact spacing of
inner-race and ball windows equals 1/BPFI and 1/(2 BSF), imbalance scales
with speed squared, misalignment peaks at 2x, the ADC quantises to its step.
The SKF 6205 geometry reproduces the published CWRU orders (3.5848, 5.4152,
2.3568, 0.3983).

| Setting | Value |
|---|---|
| Sample rate, window | 12 kHz, 2048 samples (171 ms) |
| Classes | healthy, imbalance, misalignment, outer race, inner race, ball (balanced) |
| Speed | uniform 900 to 3600 rpm per window, plus up to 2 percent wander inside the window |
| Fault severity | uniform 0.05 to 1.0 (relative amplitude of the fault component) |
| SNR | uniform -6 to +15 dB white sensor noise relative to the deterministic signal |
| Train | seed 7, 400 windows per class (2400) |
| Test | seed 8, 150 windows per class (900), independent draw |
| Calibration (static int8) | 256 training windows chosen with seed 0 |

Train and test are independent draws from the same distribution. Low
severities at low speed make healthy and mild imbalance genuinely overlap
(residual imbalance is part of the healthy model), so 100 percent is not
reachable and the remaining errors are physically meaningful.

## Models

* `baseline_features`: 43 features plus `HistGradientBoostingClassifier`
  (300 iterations, learning rate 0.08, 15 leaves, L2 0.1, seed 0). It is
  given the nominal speed and the bearing geometry, which a real system has
  from the tachometer and the bearing part number.
* `cnn_torch_fp32`: `PulseCNN`, 74,220 parameters, trained 20 epochs with
  AdamW and a one-cycle schedule, batch 64, random circular shift
  augmentation, 10 percent validation split, best validation epoch kept,
  seed 0, 4 threads. The CNN receives only the raw window.
* `cnn_onnx_fp32`: the same weights exported to ONNX (opset 17).
* `cnn_onnx_int8_dynamic`: `quantize_dynamic`, int8 per-channel weights,
  activation scales computed at runtime.
* `cnn_onnx_int8_static`: `quantize_static`, QOperator format, uint8
  activations, int8 per-channel weights, MinMax calibration, restricted to
  Conv, Gemm and MaxPool. See the note on the static recipe below.

## Reproduce

```
make setup
make bench            # about four minutes on an Apple Silicon laptop
cat benchmarks/RESULTS.md
```

`python -m pulse.bench --skip-train` re-measures existing artefacts without
retraining. `--check-only` evaluates the threshold rules on an existing
`results.json`.

Determinism: the data is a pure function of the seeds. CNN training is
deterministic on one machine (seeded generator, fixed thread count,
`torch.use_deterministic_algorithms(True)`); a test trains twice and
compares every weight. Across platforms (macOS arm64 versus Linux x86 in
CI) floating point differences in BLAS make the trained weights differ
slightly, so the CI thresholds carry a margin below the numbers measured
here rather than asserting the exact values.

## Quality gate

`benchmarks/thresholds.json` holds explicit rules of the form
`variants[variant][metric] <op> value`; `python -m pulse.bench` applies them
after measuring and exits non-zero when any fails, which is what CI runs.
The rules were set after the first measurement with these margins:

* Accuracy floors sit 4 to 6 points below the measured values (CNN
  variants at 0.90, the baseline at 0.95), wide enough for cross-platform
  training differences, tight enough to catch a regression such as the
  static quantisation failure described below (0.61).
* int8 artefacts must be at most 0.35 of the fp32 ONNX size (measured 0.30).
* ONNX fp32 must agree with PyTorch on at least 99 percent of predictions
  with a maximum probability difference of 1e-3 (measured exact parity).
* int8 variants must agree with PyTorch on at least 95 percent of
  predictions.
* Latency ceilings are loose absolute sanity bounds (2 ms for the static
  int8 graph, 5 ms for fp32 ONNX) because absolute latency depends on the
  CI runner.

## A result worth knowing about

The first static int8 export quantised the whole graph, including the
in-graph standardisation, and scored 0.6056 against 0.9611 for fp32.
The squared-amplitude node inside the standardiser spans several orders
of magnitude across windows (healthy windows at 0.03 g, imbalance at
several g), so one 8-bit MinMax range crushed quiet windows to zero.
Restricting quantisation to Conv, Gemm and MaxPool restored 0.9611 with
99.78 percent agreement and no measurable latency cost. The lesson is
general: quantise the compute, not the normalisation.

## Limitations of this evaluation

* Synthetic data: the physics is motivated and tested, but no real bearing
  was harmed. A real deployment would need a domain shift study against
  measured data (CWRU, Paderborn, IMS) before any accuracy claim transfers.
* Single machine geometry and one resonance; a fleet with different
  bearings would need per-machine feature parameters (the baseline) or
  retraining (the CNN).
* Latency and memory are measured on a laptop CPU, not on an ARM Cortex-A
  or microcontroller; the ratios between variants transfer better than the
  absolute numbers.
* The health score weights and alert thresholds are reasoned defaults, not
  tuned against labelled incidents.
