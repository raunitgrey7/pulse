"""Command line entry: ``python -m pulse.bench [--check-only]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from pulse.bench.check import check
from pulse.bench.run import BenchConfig, render_table, run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pulse.bench", description="Train, measure and gate every model variant.")
    parser.add_argument("--out", type=Path, default=Path("benchmarks"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--n-train", type=int, default=400, help="training windows per class")
    parser.add_argument("--n-test", type=int, default=150, help="test windows per class")
    parser.add_argument("--thresholds", type=Path, default=Path("benchmarks/thresholds.json"))
    parser.add_argument("--check-only", action="store_true", help="only evaluate thresholds on existing results")
    parser.add_argument("--no-check", action="store_true", help="skip the threshold gate")
    parser.add_argument("--skip-train", action="store_true", help="measure existing artifacts without retraining")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    results_path = args.out / "results.json"
    if not args.check_only:
        config = BenchConfig(
            out_dir=args.out,
            artifacts_dir=args.artifacts,
            data_dir=args.data,
            n_train_per_class=args.n_train,
            n_test_per_class=args.n_test,
            epochs=args.epochs,
            skip_train=args.skip_train,
        )
        results = run_benchmark(config)
        print(render_table(results["variants"]))
        print(f"wrote {results_path} in {results['total_seconds']} s")
    if args.no_check or not args.thresholds.exists():
        return 0
    return 0 if check(results_path, args.thresholds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
