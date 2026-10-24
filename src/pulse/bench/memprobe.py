"""Peak memory and cold start of one model variant, measured in a fresh process.

Peak RSS is a property of a whole process: an int8 ONNX graph is small, but
what the device pays for is the runtime it needs. Measuring inside the
benchmark process would charge every variant for torch and scikit-learn
being imported already, so each variant is run here in its own interpreter
and only imports what it needs. The script prints one JSON object.
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

from pulse.models.runtime import ModelKind, load_predictor
from pulse.signals.dataset import Dataset


def peak_rss_mb() -> float:
    """Peak resident set size of this process in mebibytes (platform units differ)."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return raw / divisor


def probe(kind: ModelKind, path: Path, data: Path, threads: int, name: str) -> dict[str, float | int | str]:
    """Load the artefact, score every test window one at a time, report peak RSS."""
    start = time.perf_counter()
    predictor = load_predictor(kind, path, threads, name)
    load_seconds = time.perf_counter() - start
    dataset = Dataset.load(data)
    correct = 0
    for i in range(len(dataset)):
        probs = predictor.predict_proba(dataset.x[i : i + 1], dataset.rpm[i : i + 1])
        correct += int(probs.argmax(axis=1)[0] == dataset.y[i])
    return {
        "name": name,
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "load_seconds": round(load_seconds, 4),
        "accuracy": correct / len(dataset),
        "n": len(dataset),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", type=ModelKind, choices=list(ModelKind), required=True)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--name", default="variant")
    args = parser.parse_args(argv)
    result = probe(args.kind, args.path, args.data, args.threads, args.name)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
