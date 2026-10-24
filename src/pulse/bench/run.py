"""Benchmark orchestration: data, training, export, quantisation, measurement, report.

The same test set, drawn with its own seed, is used for every variant, and
the same 300 windows are timed for every variant. Results are written as
JSON (the source of truth) and as a Markdown table copied into the README.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from pulse.bench.latency import measure_latency
from pulse.models.baseline import FeatureBaseline
from pulse.models.cnn import TrainConfig, save_cnn, train_cnn
from pulse.models.export import export_onnx, preprocess_for_quantisation, quantize_dynamic_int8, quantize_static_int8
from pulse.models.runtime import ModelKind, Predictor, load_predictor
from pulse.signals.dataset import Dataset, DatasetConfig, make_dataset
from pulse.signals.generator import CLASS_NAMES, MachineSpec

RESULTS_SCHEMA_VERSION = 1
FP32_REFERENCE = "cnn_onnx_fp32"
THREAD_ENV_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
TRAINING_FACTS = "training.json"


@dataclass(frozen=True)
class BenchConfig:
    """Everything that determines the published numbers."""

    out_dir: Path = Path("benchmarks")
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")
    n_train_per_class: int = 400
    n_test_per_class: int = 150
    train_seed: int = 7
    test_seed: int = 8
    model_seed: int = 0
    epochs: int = 20
    train_threads: int = 4
    inference_threads: int = 1
    latency_windows: int = 300
    latency_warmup: int = 30
    calibration_windows: int = 256
    skip_train: bool = False


@dataclass(frozen=True)
class Variant:
    """A model artefact to measure."""

    name: str
    kind: ModelKind
    path: Path


def _environment() -> dict[str, Any]:
    import onnxruntime
    import sklearn
    import torch

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError:
        commit = ""
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "cpu_count": platform.os.cpu_count(),
        "git_commit": commit or None,
    }


def build_datasets(config: BenchConfig) -> tuple[Dataset, Dataset, Path]:
    """Draw (or load cached) train and test sets and return the test cache path."""
    spec = MachineSpec()
    train_cfg = DatasetConfig(n_per_class=config.n_train_per_class, seed=config.train_seed, spec=spec)
    test_cfg = DatasetConfig(n_per_class=config.n_test_per_class, seed=config.test_seed, spec=spec)
    train = make_dataset(train_cfg, config.data_dir)
    test = make_dataset(test_cfg, config.data_dir)
    test_path = config.data_dir / f"pulse_synth_{test_cfg.cache_key()}.npz"
    return train, test, test_path


def train_models(config: BenchConfig, train: Dataset) -> tuple[list[Variant], dict[str, Any]]:
    """Fit the baseline and the CNN, export and quantise; return variants plus training facts."""
    art = config.artifacts_dir
    art.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    baseline = FeatureBaseline(MachineSpec()).fit(train)
    baseline.save(art / "baseline.pkl")
    baseline_seconds = time.perf_counter() - start
    start = time.perf_counter()
    model, history = train_cnn(
        train, TrainConfig(epochs=config.epochs, threads=config.train_threads, seed=config.model_seed)
    )
    cnn_seconds = time.perf_counter() - start
    save_cnn(model, art / "cnn_fp32.pt")
    export_onnx(model, art / "cnn_fp32.onnx", train.x.shape[1])
    preprocess_for_quantisation(art / "cnn_fp32.onnx", art / "cnn_fp32_pre.onnx")
    quantize_dynamic_int8(art / "cnn_fp32_pre.onnx", art / "cnn_int8_dynamic.onnx")
    rng = np.random.default_rng(config.model_seed)
    calibration = train.x[rng.choice(len(train), size=config.calibration_windows, replace=False)]
    quantize_static_int8(art / "cnn_fp32_pre.onnx", art / "cnn_int8_static.onnx", calibration)
    variants = _variants(art)
    training = {
        "baseline_seconds": round(baseline_seconds, 2),
        "cnn_seconds": round(cnn_seconds, 2),
        "cnn_epochs": config.epochs,
        "cnn_best_epoch": history.best_epoch,
        "cnn_val_accuracy_per_epoch": [round(a, 4) for a in history.val_accuracy],
        "cnn_train_loss_per_epoch": [round(loss, 4) for loss in history.train_loss],
        "cnn_parameters": model.n_parameters(),
    }
    (art / TRAINING_FACTS).write_text(json.dumps(training, indent=2))
    return variants, training


def _variants(art: Path) -> list[Variant]:
    return [
        Variant("baseline_features", ModelKind.BASELINE, art / "baseline.pkl"),
        Variant("cnn_torch_fp32", ModelKind.TORCH, art / "cnn_fp32.pt"),
        Variant(FP32_REFERENCE, ModelKind.ONNX, art / "cnn_fp32.onnx"),
        Variant("cnn_onnx_int8_dynamic", ModelKind.ONNX, art / "cnn_int8_dynamic.onnx"),
        Variant("cnn_onnx_int8_static", ModelKind.ONNX, art / "cnn_int8_static.onnx"),
    ]


def reuse_models(config: BenchConfig) -> tuple[list[Variant], dict[str, Any]]:
    """Measure existing artefacts again without retraining (development aid, same numbers otherwise)."""
    variants = _variants(config.artifacts_dir)
    missing = [v.path.name for v in variants if not v.path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot reuse artefacts, missing: {missing}")
    training = json.loads((config.artifacts_dir / TRAINING_FACTS).read_text())
    return variants, training


def measure_quality(predictor: Predictor, test: Dataset) -> dict[str, Any]:
    """Accuracy, macro F1, per-class recall and the confusion matrix on the test set."""
    probs = predictor.predict_proba(test.x, test.rpm)
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(test.y, pred, labels=list(range(len(CLASS_NAMES))))
    recall = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    return {
        "accuracy": float((pred == test.y).mean()),
        "macro_f1": float(f1_score(test.y, pred, average="macro")),
        "per_class_recall": {name: float(r) for name, r in zip(CLASS_NAMES, recall, strict=True)},
        "confusion_matrix": cm.tolist(),
        "probs": probs,
    }


def measure_latency_for(predictor: Predictor, test: Dataset, config: BenchConfig) -> dict[str, float | int]:
    n = min(config.latency_windows, len(test))

    def call(i: int) -> object:
        j = i % n
        return predictor.predict_proba(test.x[j : j + 1], test.rpm[j : j + 1])

    return measure_latency(call, config.latency_warmup, n).as_dict()


def probe_memory(variant: Variant, test_path: Path, threads: int) -> dict[str, Any]:
    """Run the memory probe in a fresh interpreter with pinned thread pools and parse its JSON line."""
    cmd = [
        sys.executable,
        "-m",
        "pulse.bench.memprobe",
        "--kind",
        variant.kind.value,
        "--path",
        str(variant.path),
        "--data",
        str(test_path),
        "--threads",
        str(threads),
        "--name",
        variant.name,
    ]
    env = {**os.environ, **{var: str(threads) for var in THREAD_ENV_VARS}}
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True, env=env, timeout=1800)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _derived_metrics(results: dict[str, dict[str, Any]]) -> None:
    """Size ratios and onnx-vs-torch parity, added in place; probabilities are dropped."""
    probs = {name: row.pop("probs") for name, row in results.items()}
    ref_size = results[FP32_REFERENCE]["size_bytes"]
    torch_probs = probs["cnn_torch_fp32"]
    for name, row in results.items():
        row["size_ratio_vs_fp32"] = round(row["size_bytes"] / ref_size, 4)
        if name.startswith("cnn_onnx"):
            row["max_abs_prob_diff_vs_torch"] = float(np.abs(probs[name] - torch_probs).max())
            row["prediction_agreement_vs_torch"] = float((probs[name].argmax(1) == torch_probs.argmax(1)).mean())


def measure_variants(variants: list[Variant], test: Dataset, test_path: Path, config: BenchConfig) -> dict[str, Any]:
    """Quality, size, latency and memory for every variant."""
    import torch

    torch.set_num_threads(config.inference_threads)
    results: dict[str, dict[str, Any]] = {}
    for variant in variants:
        predictor = load_predictor(variant.kind, variant.path, config.inference_threads, variant.name)
        row: dict[str, Any] = {"artifact": str(variant.path), "kind": variant.kind.value}
        row.update(measure_quality(predictor, test))
        row["size_bytes"] = variant.path.stat().st_size
        latency = measure_latency_for(predictor, test, config)
        row.update({f"latency_{k}": v for k, v in latency.items() if k != "repeats"})
        row["latency_repeats"] = latency["repeats"]
        memory = probe_memory(variant, test_path, config.inference_threads)
        row["peak_rss_mb"] = memory["peak_rss_mb"]
        row["load_seconds"] = memory["load_seconds"]
        row["memprobe_accuracy"] = memory["accuracy"]
        results[variant.name] = row
    _derived_metrics(results)
    return results


def _dataset_facts(train: Dataset, test: Dataset, config: BenchConfig) -> dict[str, Any]:
    cfg = DatasetConfig()
    return {
        "train_windows": len(train),
        "test_windows": len(test),
        "n_samples": int(train.x.shape[1]),
        "fs_hz": cfg.spec.fs_hz,
        "classes": list(CLASS_NAMES),
        "rpm_range": list(cfg.rpm_range),
        "severity_range": list(cfg.severity_range),
        "snr_db_range": list(cfg.snr_db_range),
        "train_seed": config.train_seed,
        "test_seed": config.test_seed,
    }


def render_table(variants: dict[str, dict[str, Any]]) -> str:
    """Markdown table with the headline metrics, one row per variant."""
    header = (
        "| Variant | Accuracy | Macro F1 | Size (KB) | p50 latency (ms) | p95 latency (ms) "
        "| Peak RSS (MB) | Load (s) |\n|---|---:|---:|---:|---:|---:|---:|---:|"
    )
    rows = [
        f"| {name} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['size_bytes'] / 1024:.1f} "
        f"| {r['latency_p50_ms']:.3f} | {r['latency_p95_ms']:.3f} | {r['peak_rss_mb']:.1f} | {r['load_seconds']:.3f} |"
        for name, r in variants.items()
    ]
    return "\n".join([header, *rows])


def run_benchmark(config: BenchConfig) -> dict[str, Any]:
    """Execute the whole pipeline and write results.json plus RESULTS.md."""
    started = time.perf_counter()
    train, test, test_path = build_datasets(config)
    variants, training = reuse_models(config) if config.skip_train else train_models(config, train)
    measured = measure_variants(variants, test, test_path, config)
    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in asdict(config).items()},
        "dataset": _dataset_facts(train, test, config),
        "training": training,
        "variants": measured,
        "total_seconds": round(time.perf_counter() - started, 2),
    }
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    table = render_table(measured)
    stamp = f"Generated {results['generated_at']} on {results['environment']['platform']}"
    (config.out_dir / "RESULTS.md").write_text(f"# Benchmark results\n\n{stamp}\n\n{table}\n")
    return results
