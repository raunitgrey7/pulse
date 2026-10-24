# Report

Project: pulse, edge AI for predictive maintenance. Built on macOS-26.5.1-arm64-arm-64bit-Mach-O with
Python 3.13.17, torch 2.14.1 (training only), onnxruntime 1.31.0,
numpy 2.5.4, scikit-learn 1.9.1; 15 CPUs. Benchmark results in
this file were generated at 2026-10-24T21:53:54+00:00 at commit b3d4ae5 by `make bench`.

## What was built

* `pulse.signals`: a physically motivated vibration generator for a motor-driven shaft on ball
  bearings. Bearing kinematics (BPFO, BPFI, BSF, FTF) from geometry; impact trains at the
  kinematic frequencies ringing a damped structural resonance, with load-zone modulation for
  inner-race faults and cage modulation for ball faults, slip jitter, imbalance scaling with speed
  squared, misalignment at 2x, speed wander on an integrated phase, resonance-coloured background,
  mains hum, white sensor noise at a target SNR and a 16-bit ADC. Balanced seeded datasets cached by
  configuration hash. The SKF 6205 geometry reproduces the published CWRU orders.
* `pulse.features`: 43 features (time statistics, 16 log band energies, spectral shape, shaft-order
  amplitudes, envelope-spectrum amplitudes at the fault frequencies and sidebands, band kurtosis).
* `pulse.models`: features plus histogram gradient boosting; `PulseCNN`
  (74,220 parameters) with in-graph standardisation and a wide receptive field; ONNX
  export (opset 17, dynamic batch); dynamic and static int8 quantisation with onnxruntime; one
  `Predictor` interface over all artefacts.
* `pulse.bench`: trains, exports, quantises and measures every variant on the same seeded test set,
  including peak RSS and load time of a fresh interpreter per variant, parity against PyTorch, a
  Markdown table and a threshold gate used by CI.
* `pulse.registry`: Ed25519-signed release manifests verified by signer, signature and content
  hash; staged rollout (draft 0, canary 5, pilot 25, fleet 100 percent, rollback) with stable device
  buckets and a canary guard (minimum reports, error rate, fault-rate ratio against the incumbent).
* `pulse.health`: 0 to 100 health score (smoothed fault probability, RMS versus a per-device
  baseline graded like ISO 10816 zones, impulsiveness, trend) and alert rules with deduplication,
  acknowledgement, automatic resolution and sinks.
* `pulse.api`: FastAPI ingest and control plane with SQLite, strict Pydantic validation, API key
  auth, request ids, JSON logs, Prometheus text metrics, registry endpoints and a vanilla JS
  dashboard.
* `pulse.fleet`: simulated devices that run the deployed ONNX model locally, report summaries,
  and only load releases that verify against their trust key; a runner that drives them against
  any `httpx.Client`.
* Packaging: Makefile (`setup`, `lint`, `test`, `bench`, `serve`, `fleet`), multi-stage non-root
  Dockerfile without PyTorch, docker-compose with API and fleet, GitHub Actions CI with lint, tests
  and the benchmark gate.

## Measured results

Command: `make bench` (equivalent to `uv run python -m pulse.bench`). Total 80.1 s.
Baseline training 21.92 s; CNN training 49.64 s for 20 epochs with the best
validation epoch 18 kept (validation accuracy per epoch:
0.375, 0.558, 0.617, 0.667, 0.571, 0.567, 0.858, 0.875, 0.954, 0.871, 0.963, 0.958, 0.971, 0.963, 0.975, 0.971, 0.967, 0.975, 0.975, 0.971).
Data: 2400 training windows (seed 7), 900 test windows (seed 8),
2048 samples at 12000 Hz, speed 900 to 3600 rpm, severity 0.05 to 1.0,
SNR -6 to 15 dB. Latency: batch size one, one intra-op thread, 30 warm-up calls,
300 timed windows. Peak RSS and load time: fresh interpreter per variant with
`OMP_NUM_THREADS=1`.

| Variant | Accuracy | Macro F1 | Size (KB) | p50 latency (ms) | p95 latency (ms) | Peak RSS (MB) | Load (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_features | 0.9900 | 0.9900 | 2038.9 | 4.261 | 4.573 | 139.6 | 0.478 |
| cnn_torch_fp32 | 0.9611 | 0.9612 | 302.5 | 0.240 | 0.253 | 206.5 | 0.322 |
| cnn_onnx_fp32 | 0.9611 | 0.9612 | 294.1 | 0.153 | 0.159 | 66.6 | 0.016 |
| cnn_onnx_int8_dynamic | 0.9600 | 0.9601 | 89.1 | 0.101 | 0.109 | 66.6 | 0.017 |
| cnn_onnx_int8_static | 0.9611 | 0.9612 | 88.1 | 0.066 | 0.067 | 67.1 | 0.017 |

Parity and size ratios (from `results.json`):

| Variant | size ratio vs fp32 ONNX | prediction agreement vs torch | max abs probability diff vs torch | accuracy at batch size one (memory probe) |
|---|---:|---:|---:|---:|
| baseline_features | 6.9333 | n/a | n/a | 0.9900 |
| cnn_torch_fp32 | 1.0288 | n/a | n/a | 0.9611 |
| cnn_onnx_fp32 | 1.0000 | 1.0000 | 2.00e-06 | 0.9611 |
| cnn_onnx_int8_dynamic | 0.3031 | 0.9989 | 6.67e-02 | 0.9611 |
| cnn_onnx_int8_static | 0.2994 | 0.9978 | 7.28e-02 | 0.9611 |

Per-class recall:

* baseline_features: healthy 0.993, imbalance 0.953, misalignment 1.000, outer_race 1.000, inner_race 0.993, ball 1.000
* cnn_onnx_fp32: healthy 0.993, imbalance 0.920, misalignment 0.967, outer_race 0.960, inner_race 0.960, ball 0.967
* cnn_onnx_int8_static: healthy 0.993, imbalance 0.920, misalignment 0.967, outer_race 0.960, inner_race 0.967, ball 0.960

Confusion matrix of cnn_onnx_int8_static (rows true, columns predicted, order healthy, imbalance, misalignment, outer_race, inner_race, ball):

```
 149    1    0    0    0    0
  11  138    1    0    0    0
   0    5  145    0    0    0
   2    0    0  144    1    3
   1    0    0    1  145    3
   2    0    0    2    2  144
```

The residual errors are physically meaningful: 11 of 150 imbalance windows are read as healthy
(low severity at low speed, where the imbalance force is smallest and overlaps the residual
imbalance of a healthy machine), and 5 misalignment windows as imbalance. The baseline makes the
same kind of error (7 imbalance windows read as healthy) but fewer, because it is given the
speed and the bearing geometry.

Threshold gate: all 13 rules in `benchmarks/thresholds.json` passed (see the `make bench` tail below).

### Tail of `make lint`

```
uv run ruff check .
All checks passed!
uv run ruff format --check .
59 files already formatted
exit 0
```

### Tail of `make test`

```
........................................................................ [ 87%]
....................                                                     [100%]
164 passed in 5.25s
exit 0
```

164 tests: unit, integration (API through `TestClient`, fleet runner against the real app),
property-based with hypothesis (bearing identities, feature scale invariance, softmax, health score
bounds, version parsing, device buckets), adversarial (tampered model bytes, forged manifest fields,
wrong signer, NaN and out-of-range telemetry, oversized batches, foreign pickles) and determinism
(same seed gives identical windows, datasets, features and trained CNN weights).

### Tail of `make bench`

```
wrote benchmarks/results.json in 80.1 s
PASS baseline_features.accuracy = 0.99 (want >= 0.95)
PASS cnn_torch_fp32.accuracy = 0.961111 (want >= 0.9)
PASS cnn_onnx_fp32.accuracy = 0.961111 (want >= 0.9)
PASS cnn_onnx_fp32.prediction_agreement_vs_torch = 1 (want >= 0.99)
PASS cnn_onnx_fp32.max_abs_prob_diff_vs_torch = 1.99676e-06 (want <= 0.001)
PASS cnn_onnx_fp32.latency_p50_ms = 0.153313 (want <= 5)
PASS cnn_onnx_int8_dynamic.accuracy = 0.96 (want >= 0.9)
PASS cnn_onnx_int8_dynamic.size_ratio_vs_fp32 = 0.3031 (want <= 0.35)
PASS cnn_onnx_int8_dynamic.prediction_agreement_vs_torch = 0.998889 (want >= 0.95)
PASS cnn_onnx_int8_static.accuracy = 0.961111 (want >= 0.9)
PASS cnn_onnx_int8_static.size_ratio_vs_fp32 = 0.2994 (want <= 0.35)
PASS cnn_onnx_int8_static.prediction_agreement_vs_torch = 0.997778 (want >= 0.95)
PASS cnn_onnx_int8_static.latency_p50_ms = 0.06575 (want <= 2)
13/13 threshold rules passed
bench exit 0
```

### Live run over HTTP

Commands (port 8765, `APP_API_KEY=demo-key`, fresh database and registry):
`pulse serve`, `pulse keygen`, `pulse release --model artifacts/cnn_int8_static.onnx --version 1.0.0 ...`,
`pulse promote --version 1.0.0`, `pulse fleet --devices 12 --rounds 40 --interval 0.02 --trust-key <public key>`.
Observed: publish 201, promote to canary (5 percent), unauthenticated request 401, fleet summary
`rounds 40, reports_sent 480, reports_accepted 480, alerts_opened 19, model_updates 1,
rejected_releases 0, post_failures 0` in 1.6 s wall time; the one device in the canary bucket
(dev-008) pulled, verified and ran release 1.0.0; 6 of 12 devices critical with dominant classes
imbalance, misalignment, outer_race, inner_race, ball, imbalance (the six devices with degradation
plans) and 6 healthy; 16 open alerts across `health_critical`, `fault_persistent` and
`vibration_zone_d`; a second promotion refused with `only 40 reports, need 50`; dashboard 200;
`/metrics` showed per-class report counters, band gauges, `pulse_fleet_health_mean 60.2317` and one
canary release with every other stage at 0; 125 JSON log lines with 40 `202` ingest responses.

### Docker

`docker build -t pulse:local .` succeeded (176 MB compressed content, 796 MB on disk; no PyTorch).
`docker run -p 8766:8000 pulse:local`: `/health` 200, dashboard 200, process runs as uid 999
(`pulse`), `onnxruntime 1.31.0` present, `torch` absent. `docker compose config` validates.

## Findings recorded during the build

1. Receptive field decides inner-race versus ball. The first CNN (four blocks, 9 ms receptive
   field, global average pooling) scored 0.8117 on a 600-window development test set with inner race
   and ball at roughly 50 percent each; widening the receptive field to most of the window and
   flattening the last 16 frames gave 0.9667 (GAP head 0.955) on the same set. Those two classes
   differ by modulation at shaft versus cage rate, which spans 30 to 170 ms.
2. Quantise the compute, not the normalisation. Static int8 over the whole graph scored 0.6056;
   restricting it to Conv, Gemm and MaxPool gave 0.9611, identical to fp32.
3. Dynamic quantisation is batch dependent: activation scales are computed per batch, so the batched
   evaluation (0.9600) and single-window evaluation (0.9611) differ by one window.
   Static quantisation gives the same answer either way.
4. Pin thread pools before measuring. scikit-learn's boosted-tree predict opens an OpenMP region per
   tree; unpinned on 15 cores, 900 single-window predictions ran for more than 35 CPU-minutes without
   finishing. With `threadpoolctl` and `OMP_NUM_THREADS=1` the same probe takes about a second.
5. The dynamo ONNX exporter (torch 2.14) produces a graph that fails onnxruntime's shape inference
   inside the quantiser (Gemm weight annotation); the TorchScript exporter is used instead.
6. An adversarial probe found that a NaN in a request body crashed FastAPI's default validation
   handler while echoing the input; a sanitising handler now returns a clean 422.
7. The live run found two defects the unit tests had not: `pulse fleet --api ...` through the CLI
   exited immediately (argparse REMAINDER does not capture leading options) and the
   `pulse_registry_releases` gauge kept stale stage counts. Both fixed with regression tests.

## Known gaps and what was not achieved

* Synthetic data only; no evaluation on recorded bearing data (CWRU, Paderborn, IMS). No claim
  here transfers to a real machine without that study.
* One machine geometry and one resonance frequency.
* Latency and memory were measured on an Apple Silicon laptop, not on an embedded ARM part.
* Browser tools were not used, so the dashboard was verified by serving it (HTTP 200), a node syntax
  check of `app.js` and exercising every endpoint it calls, not by rendering it.
* The CI workflow was written but not executed (nothing was pushed, by instruction). Its benchmark
  gate carries margins because CNN training is deterministic per machine, not across platforms.
* Alerts live in process memory; the stale-device check is an endpoint (`POST /v1/alerts/check-stale`),
  not a scheduler. A multi-instance deployment would need a shared store for both.
* Health score weights and alert thresholds are reasoned defaults, not tuned on labelled incidents.
* The Docker image is 796 MB on disk even without PyTorch (scipy, scikit-learn, onnx, onnxruntime);
  a device image could drop scikit-learn and scipy if it only runs ONNX models.
* `docker compose up` was not run end to end; the compose file was validated with
  `docker compose config` and the same image was run and probed directly.
