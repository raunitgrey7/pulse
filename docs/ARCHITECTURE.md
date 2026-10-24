# Architecture

Pulse is organised as a pipeline from physics to fleet operations. Each
layer has one owner module and a narrow interface, so the model that runs
on a device can be swapped without touching the API, and the physics can be
tuned without touching the models.

```
 physics                 learning                       compression             operations
+----------------+     +--------------------+     +---------------------+     +-------------------------+
| signals/       |     | features.py        |     | models/export.py    |     | fleet/  (devices)       |
|  bearing.py    | --> | models/baseline.py | --> |  ONNX fp32          | --> |  sense, infer, report   |
|  generator.py  |     | models/cnn.py      |     |  int8 dynamic       |     |  verify + load releases |
|  dataset.py    |     |                    |     |  int8 static        |     +-----------+-------------+
+----------------+     +--------------------+     +---------------------+                 | HTTP (summaries)
        |                        |                          |                              v
        |                        v                          v                 +-------------------------+
        |              bench/ (measures every variant: accuracy, size,        | api/  FastAPI           |
        |                      latency, peak RSS; writes results.json)        |  ingest -> store (SQLite)|
        |                                                                     |  health/ scoring, alerts |
        +----- same generator drives the simulated devices ------------------>|  registry/ signed rollout|
                                                                              |  /metrics, dashboard     |
                                                                              +-------------------------+
```

## Modules

### `pulse.signals`
* `bearing.py`: bearing geometry and the BPFO, BPFI, BSF and FTF formulas.
  Shared by the generator and the feature extractor so both agree on where
  a fault should appear.
* `generator.py`: one accelerometer window from an operating point
  (`WindowParams`: condition, rpm, severity, SNR, speed drift, load
  modulation, slip). Shaft harmonics are built on an integrated phase so
  speed wander is honoured; bearing defects are impact trains at the
  kinematic frequency ringing a damped structural resonance; inner-race and
  ball impacts are amplitude modulated at shaft and cage rate; a background
  of resonance-coloured noise, mains hum, white sensor noise at the target
  SNR and a 16-bit ADC complete the sensor chain.
* `dataset.py`: balanced labelled draws with independent operating points
  per window, cached under `data/` by a hash of the configuration and a
  generator version.

### `pulse.features`
Forty-three hand-crafted features: time statistics, sixteen log band
energies, spectral shape, shaft-order amplitudes at the known speed, and
envelope-spectrum amplitudes at the bearing fault frequencies after
band-passing the resonance and demodulating with the Hilbert transform.

### `pulse.models`
* `baseline.py`: features plus `HistGradientBoostingClassifier`, pickled as
  one object so extractor and classifier cannot drift apart.
* `cnn.py`: `PulseCNN`, a 74k-parameter 1D CNN. Standardisation is inside
  the graph and the removed log scale is fed to the head. A wide strided
  first convolution acts as a filter bank; rectification and pooling turn
  it into an envelope detector; four more blocks see the modulation at a
  low frame rate; the head flattens the last 16 frames so modulation that
  spans the window stays visible. Training is deterministic (seeded, fixed
  threads, deterministic algorithms) and keeps the best validation epoch.
* `export.py`: ONNX export (TorchScript exporter, opset 17, dynamic batch),
  onnxruntime pre-processing, dynamic int8 and static int8 (QOperator,
  per-channel, MinMax calibration on 256 training windows, restricted to
  Conv, Gemm and MaxPool so the normalisation stays in float).
* `runtime.py`: `Predictor` protocol with adapters for the pickled baseline,
  the PyTorch model and any ONNX graph. Heavy imports are local so an ONNX
  device never imports torch.

### `pulse.bench`
`run.py` draws the data, trains, exports, quantises and measures every
variant on the same test set: accuracy, macro F1, per-class recall,
confusion matrix, artefact size, single-window latency (p50, p95, mean,
min) with one intra-op thread, and the peak RSS and load time of a fresh
interpreter (`memprobe.py`) with pinned thread pools. `check.py` applies
explicit threshold rules from `benchmarks/thresholds.json` and is the CI
quality gate.

### `pulse.registry`
* `signing.py`: Ed25519 keys; publishers hold the seed, devices hold the
  32-byte public key.
* `releases.py`: the manifest (version, hash, size, kind, metrics) signed
  over its canonical JSON, and `verify_release`, which checks signer,
  signature and content hash.
* `rollout.py`: stages draft (0 percent), canary (5), pilot (25), fleet
  (100), rolled back (0). Devices are bucketed by a hash of their id so the
  canary cohort is stable. Promotion past canary requires `CanaryGuard` to
  approve: enough reports, error rate under 5 percent, and a fault rate no
  more than twice the incumbent release's. State is JSON on disk next to
  the model bytes.

### `pulse.health`
* `scoring.py`: 0 to 100 score from an EWMA of the fault probability, the
  RMS level relative to a per-device baseline graded like ISO 10816 zones,
  impulsiveness from kurtosis and the RMS trend. Bands: healthy at or above
  70, watch from 40, critical below.
* `alerts.py`: rules (`health_critical`, `fault_persistent`,
  `vibration_zone_d`, `device_stale`), one open alert per device and rule,
  occurrence counts, acknowledgement, automatic resolution, pluggable sinks
  (log, optional webhook).

### `pulse.api`
FastAPI application. `service.py` owns the ingest pipeline: store the
report, score the device on its last ten reports, persist the score, run
the rules, update metrics. `store.py` is SQLite (WAL) with tables for
devices, telemetry and scores; a device's RMS baseline is the median of its
first twenty reports and never moves afterwards. `routes/telemetry.py` and
`routes/registry.py` expose the endpoints; `logging.py` emits one JSON line
per request with a request id; `metrics.py` renders Prometheus text without
a client library; `auth.py` enforces `X-API-Key` when `APP_API_KEY` is set.
The dashboard under `static/` is plain HTML, CSS and JavaScript.

### `pulse.fleet`
`device.py` is the edge loop: generate a window for the device's current
condition (healthy until an onset step, then a ramping fault), run the
local ONNX model, summarise the window and build a `TelemetryReport`.
`sync_model` asks the registry for the device's assignment and only loads
a release that verifies against the device's trust key. `runner.py` drives
many devices in rounds against any `httpx.Client`, which is how the tests
run the whole loop in-process.

## Data flow for one report

1. Device draws a window, runs the ONNX model (about 0.1 ms single-threaded
   for the static int8 graph), computes RMS, kurtosis and crest factor.
2. `POST /v1/telemetry` validates the batch (finite numbers, plausible
   ranges, a probability vector that sums to one, capped batch size).
3. `Services.ingest` inserts the report, fixes the baseline when the
   twentieth report arrives, scores the device, persists the score.
4. `AlertRules.evaluate` looks at the last ten scores and predictions;
   `AlertManager.apply` opens, refreshes or resolves alerts and fans new
   ones out to the sinks.
5. Metrics counters and gauges update; the dashboard polls the read
   endpoints every three seconds.

## Deployment

One container runs the API with a volume for `data/` (SQLite and the
registry). The same image runs the fleet simulator pointed at the API. The
image deliberately excludes PyTorch: training and export are done on a
workstation with `uv sync --all-extras`, and only ONNX artefacts travel to
devices through signed releases.
