# Benchmark results

Generated 2026-10-24T21:53:54+00:00 on macOS-26.5.1-arm64-arm-64bit-Mach-O

| Variant | Accuracy | Macro F1 | Size (KB) | p50 latency (ms) | p95 latency (ms) | Peak RSS (MB) | Load (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_features | 0.9900 | 0.9900 | 2038.9 | 4.261 | 4.573 | 139.6 | 0.478 |
| cnn_torch_fp32 | 0.9611 | 0.9612 | 302.5 | 0.240 | 0.253 | 206.5 | 0.322 |
| cnn_onnx_fp32 | 0.9611 | 0.9612 | 294.1 | 0.153 | 0.159 | 66.6 | 0.016 |
| cnn_onnx_int8_dynamic | 0.9600 | 0.9601 | 89.1 | 0.101 | 0.109 | 66.6 | 0.017 |
| cnn_onnx_int8_static | 0.9611 | 0.9612 | 88.1 | 0.066 | 0.067 | 67.1 | 0.017 |
