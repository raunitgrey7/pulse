"""Command line entry: ``python -m pulse.fleet --api http://localhost:8000``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import httpx

from pulse.api.logging import configure_logging
from pulse.fleet.runner import FleetConfig, FleetRunner, build_fleet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pulse.fleet", description="Stream simulated device telemetry to the API.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--devices", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--fault-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/cnn_int8_static.onnx"), help="fallback ONNX model"
    )
    parser.add_argument("--trust-key", default=None, help="hex public key the devices accept releases from")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging()
    config = FleetConfig(
        n_devices=args.devices,
        fault_fraction=args.fault_fraction,
        seed=args.seed,
        rounds=args.rounds,
        interval_s=args.interval,
        fallback_model=args.model if args.model.exists() else None,
        trust_key_hex=args.trust_key,
    )
    if config.fallback_model is None:
        logging.getLogger("pulse.fleet").warning(
            "fallback model not found; devices start without a model", extra={"path": str(args.model)}
        )
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    with httpx.Client(base_url=args.api, headers=headers, timeout=10.0) as client:
        stats = FleetRunner(build_fleet(config), client, config).run()
    print(json.dumps(stats.as_dict()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
