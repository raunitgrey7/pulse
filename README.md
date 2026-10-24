# pulse

Edge AI for predictive maintenance of rotating machinery: a physics-based
vibration data generator with exact ground truth, a classical baseline and a
small 1D CNN, ONNX export with int8 quantisation, a benchmark that measures
accuracy, size, latency and memory for every variant, and the fleet side
that turns a model into an operated system: simulated devices streaming
telemetry, an ingest API, a model registry with Ed25519-signed releases and
staged rollout gated on canary health, per-device health scoring, alerting
and a dashboard.

Everything runs on a laptop CPU with no API key and no network at runtime.
Python 3.13, `uv`, PyTorch (training only), onnxruntime, FastAPI, SQLite.

## Why

Bearing and alignment faults are the bulk of unplanned downtime in motor
driven equipment, and the signatures are well understood physics: impacts at
kinematic frequencies, harmonics of the shaft speed. Public datasets are
small and recorded at a handful of fixed speeds, so they cannot tell you how
a model behaves across the operating envelope. Pulse generates windows from
an explicit mechanical model (bearing geometry, speed wander, structural
resonance, load-zone modulation, sensor noise, ADC) so difficulty is
controllable and labels are exact, then measures what compression does to a
model that has to run on a device, and ships the operational plumbing
(signed releases, canary rollout, health scores) that decides whether a
model is actually safe to roll out.

## Architecture

```
  signals/ (physics)        models/                   bench/                 fleet + api
  bearing kinematics  -->   features + HGB baseline   accuracy, size,        devices -> POST /v1/telemetry
  impact trains, speed      PulseCNN (74k params)     p50/p95 latency,       store (SQLite) -> health score
  wander, resonance,  -->   ONNX fp32 export     -->  peak RSS, parity  -->  -> alert rules -> dashboard
  noise, 16-bit ADC         int8 dynamic / static     results.json           registry: signed releases,
                                                                             canary -> pilot -> fleet
```

See `docs/ARCHITECTURE.md` for modules and data flow and `docs/EVALUATION.md`
for the methodology.

## Quickstart

```
make setup                      # uv sync with the training extra and dev tools
make bench                      # draw data, train, export, quantise, measure, gate  (about 4 min)
make serve                      # API and dashboard on http://localhost:8000
make fleet                      # in a second shell: 12 simulated devices stream for 60 s
uv run pulse keygen             # then sign and publish a release, see "Model rollout" below
```

Open http://localhost:8000 for the dashboard. `make lint` and `make test`
run ruff and the test suite.

## Measured results

Produced by `make bench` on this machine (Apple M-series, macOS, one
intra-op thread for inference, 900 held-out test windows, seeds 7 and 8).
The table is copied from `benchmarks/RESULTS.md`; the full detail including
per-class recall, confusion matrices and the environment is in
`benchmarks/results.json`.

<!-- RESULTS:BEGIN -->
| Variant | Accuracy | Macro F1 | Size (KB) | p50 latency (ms) | p95 latency (ms) | Peak RSS (MB) | Load (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_features | 0.9900 | 0.9900 | 2038.9 | 4.261 | 4.573 | 139.6 | 0.478 |
| cnn_torch_fp32 | 0.9611 | 0.9612 | 302.5 | 0.240 | 0.253 | 206.5 | 0.322 |
| cnn_onnx_fp32 | 0.9611 | 0.9612 | 294.1 | 0.153 | 0.159 | 66.6 | 0.016 |
| cnn_onnx_int8_dynamic | 0.9600 | 0.9601 | 89.1 | 0.101 | 0.109 | 66.6 | 0.017 |
| cnn_onnx_int8_static | 0.9611 | 0.9612 | 88.1 | 0.066 | 0.067 | 67.1 | 0.017 |
<!-- RESULTS:END -->

How to read it:

* The baseline is given the shaft speed and the bearing geometry, which is
  why it is the most accurate; it pays for that with 2 MB of trees and
  feature extraction (FFT, band-pass, Hilbert) that dominates its latency.
* The CNN sees only the raw window. Exporting it to ONNX changes nothing in
  accuracy (prediction agreement 1.0, max probability difference 2e-6) and
  removes PyTorch from the device: peak memory of a fresh process falls from
  206.5 MB to 66.6 MB and load time from 0.322 s to 0.016 s.
* Static int8 is the smallest and fastest variant (88.1 KB, 0.066 ms p50,
  2.3x faster than fp32 ONNX and 65x faster than the baseline) at the same
  accuracy as the float model, but only after restricting quantisation to
  the compute operators. Quantising the in-graph normalisation too cost 35
  points of accuracy (0.6056); the failure and the fix are recorded in
  `docs/EVALUATION.md`.
* Dynamic int8 agrees with the float model on 99.9 percent of windows. Its
  activation scales depend on the batch, so batched evaluation (0.9600) and
  single-window evaluation in the memory probe (0.9611) differ by one window.

## Model rollout

```
uv run pulse keygen                                  # prints seed_hex and public_key_hex
export PULSE_VERIFY_KEY_HEX=<public_key_hex>         # the API only accepts releases from this key
make serve
uv run pulse release --model artifacts/cnn_int8_static.onnx --version 1.0.0 \
    --seed-hex <seed_hex> --api http://127.0.0.1:8000
uv run pulse promote --version 1.0.0                 # draft -> canary (5 percent of devices)
uv run pulse fleet --api http://127.0.0.1:8000 --trust-key <public_key_hex> --rounds 120
uv run pulse promote --version 1.0.0                 # canary -> pilot, allowed only if the canary cohort is healthy
```

Devices verify the Ed25519 signature and the SHA-256 of every release before
loading it and keep their current model otherwise. Promotion past canary is
refused until the cohort has at least 50 reports, an error rate under 5
percent and a fault rate no more than twice the incumbent's; `rollback`
drops a release to zero percent immediately.

## API

| Method and path | Purpose |
|---|---|
| `GET /health`, `GET /metrics` | liveness; Prometheus text format |
| `POST /v1/telemetry` | batch of device reports (validated, 202) |
| `GET /v1/devices`, `GET /v1/devices/{id}` | fleet and device detail with health and recent scores |
| `GET /v1/alerts?open=true`, `POST /v1/alerts/{id}/ack` | alerts and acknowledgement |
| `GET /v1/fleet/summary` | counts by band, mean score, versions |
| `POST /v1/registry/releases`, `GET /v1/registry/releases[/{v}[/model]]` | publish (signature checked) and read releases |
| `POST /v1/registry/releases/{v}/promote`, `.../rollback` | staged rollout |
| `GET /v1/registry/assignment/{device_id}` | which release a device should run |

All `/v1` routes require `X-API-Key` when `APP_API_KEY` is set. Every
response carries `X-Request-ID`; logs are JSON lines with that id.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `APP_API_KEY` | unset | API key for `/v1` routes |
| `PULSE_DB_PATH` | `data/pulse.db` | SQLite file |
| `PULSE_REGISTRY_DIR` | `data/registry` | release files |
| `PULSE_VERIFY_KEY_HEX` | unset | pin the publisher key accepted by the registry |
| `PULSE_ALERT_WEBHOOK_URL` | unset | POST new alerts as JSON |
| `PULSE_STALE_AFTER_S` | `60` | staleness threshold for `check-stale` |

## Docker

```
docker compose up --build        # api on :8000 plus a 12-device fleet; mounts ./artifacts read-only
```

The image is multi-stage, runs as a non-root user and excludes PyTorch:
devices and the API need only onnxruntime.

## Limitations

* Synthetic data only. The physics is documented and tested, but no claim
  here transfers to a real machine without a domain shift study on measured
  data.
* One machine geometry and one resonance frequency; other bearings need
  their geometry in the feature extractor or a retrained CNN.
* Latency and memory are measured on a laptop CPU, not on an embedded ARM
  part; ratios between variants transfer better than absolute numbers.
* The health score weights and alert thresholds are reasoned defaults, not
  tuned on labelled incidents.
* Alerts and rollout state live in process memory and local files; a
  multi-instance deployment would need a shared store.
* The CI benchmark gate uses margins below the numbers measured here because
  training is deterministic per machine, not bit-identical across platforms.

## Roadmap

* Validate on CWRU and Paderborn recordings and report the shift.
* Order tracking (angular resampling) so the CNN can use the tachometer.
* Quantisation-aware training and a per-channel int8 comparison on ARM.
* Drift detection on the feature distribution per device.
* Persist alerts to SQLite and add a stale-device scheduler.

## Layout

```
src/pulse/signals     physics: bearing kinematics, generator, datasets
src/pulse/features.py hand-crafted features (time, spectrum, orders, envelope)
src/pulse/models      baseline, PulseCNN, ONNX export and quantisation, runtime adapters
src/pulse/bench       benchmark harness, memory probe, threshold gate
src/pulse/registry    Ed25519 signing, release manifests, staged rollout
src/pulse/health      health scoring, alert rules and lifecycle
src/pulse/api         FastAPI app, SQLite store, metrics, logging, dashboard
src/pulse/fleet       simulated devices and the fleet runner
tests/                unit, integration, property-based (hypothesis), adversarial, determinism
benchmarks/           results.json, RESULTS.md, thresholds.json
docs/                 ARCHITECTURE.md, EVALUATION.md
```
