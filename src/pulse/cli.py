"""The ``pulse`` command: bench, serve, fleet, keygen, release and generate.

One entry point so the README quickstart stays short. Each subcommand
delegates to the module that owns the logic; nothing here is more than
argument parsing and wiring.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx


def _cmd_keygen(_: argparse.Namespace) -> int:
    from pulse.registry.signing import SigningKey

    key = SigningKey.generate()
    print(
        json.dumps(
            {
                "seed_hex": key.to_seed().hex(),
                "public_key_hex": key.verify_key().to_hex(),
                "fingerprint": key.verify_key().fingerprint,
            }
        )
    )
    return 0


def _cmd_release(args: argparse.Namespace) -> int:
    """Sign a model file and publish it to a running API (or print the manifest)."""
    from pulse.registry.releases import sign_release
    from pulse.registry.signing import SigningKey

    key = SigningKey.from_seed(bytes.fromhex(args.seed_hex))
    model_bytes = Path(args.model).read_bytes()
    metrics: dict[str, float] = {}
    if args.metrics_from and Path(args.metrics_from).exists():
        row = json.loads(Path(args.metrics_from).read_text())["variants"].get(args.variant, {})
        metrics = {k: float(row[k]) for k in ("accuracy", "macro_f1", "latency_p50_ms", "peak_rss_mb") if k in row}
    manifest = sign_release(key, args.version, Path(args.model).name, "onnx", model_bytes, metrics, args.notes)
    body = {
        "manifest": asdict(manifest),
        "model_b64": base64.b64encode(model_bytes).decode(),
        "signer_public_key_hex": key.verify_key().to_hex(),
    }
    if not args.api:
        print(manifest.to_json())
        return 0
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    response = httpx.post(f"{args.api}/v1/registry/releases", json=body, headers=headers, timeout=30.0)
    print(json.dumps({"status": response.status_code, "body": response.json()}))
    return 0 if response.status_code == 201 else 1


def _cmd_promote(args: argparse.Namespace) -> int:
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    action = "rollback" if args.rollback else "promote"
    payload = {"reason": args.rollback} if args.rollback else None
    response = httpx.post(
        f"{args.api}/v1/registry/releases/{args.version}/{action}", json=payload, headers=headers, timeout=30.0
    )
    print(json.dumps({"status": response.status_code, "body": response.json()}))
    return 0 if response.is_success else 1


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("pulse.api.app:create_app", factory=True, host=args.host, port=args.port, log_config=None)
    return 0


def _cmd_generate(args: argparse.Namespace) -> int:
    from pulse.signals.dataset import DatasetConfig, make_dataset

    dataset = make_dataset(DatasetConfig(n_per_class=args.n_per_class, seed=args.seed))
    dataset.save(Path(args.out))
    print(json.dumps({"windows": len(dataset), "n_samples": int(dataset.x.shape[1]), "out": str(args.out)}))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from pulse.bench.__main__ import main

    return main(args.rest)


def _cmd_fleet(args: argparse.Namespace) -> int:
    from pulse.fleet.__main__ import main

    return main(args.rest)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pulse", description="Edge AI for predictive maintenance.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("keygen", help="generate an Ed25519 release signing key").set_defaults(func=_cmd_keygen)
    rel = sub.add_parser("release", help="sign a model and publish it to the registry API")
    rel.add_argument("--model", required=True)
    rel.add_argument("--version", required=True)
    rel.add_argument("--seed-hex", required=True, help="32-byte signing seed as hex (from keygen)")
    rel.add_argument("--api", default=None, help="API base URL; omit to print the manifest only")
    rel.add_argument("--api-key", default=None)
    rel.add_argument("--metrics-from", default="benchmarks/results.json")
    rel.add_argument("--variant", default="cnn_onnx_int8_static")
    rel.add_argument("--notes", default="")
    rel.set_defaults(func=_cmd_release)
    pro = sub.add_parser("promote", help="promote (or roll back) a release through the rollout ladder")
    pro.add_argument("--version", required=True)
    pro.add_argument("--api", default="http://127.0.0.1:8000")
    pro.add_argument("--api-key", default=None)
    pro.add_argument("--rollback", default=None, metavar="REASON", help="roll back instead, with a reason")
    pro.set_defaults(func=_cmd_promote)
    srv = sub.add_parser("serve", help="run the ingest API and dashboard")
    srv.add_argument("--host", default="127.0.0.1")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(func=_cmd_serve)
    gen = sub.add_parser("generate", help="write a synthetic dataset to an .npz file")
    gen.add_argument("--out", default="data/sample.npz")
    gen.add_argument("--n-per-class", type=int, default=50)
    gen.add_argument("--seed", type=int, default=0)
    gen.set_defaults(func=_cmd_generate)
    for name, func, help_text in (
        ("bench", _cmd_bench, "run the benchmark"),
        ("fleet", _cmd_fleet, "run the device fleet"),
    ):
        p = sub.add_parser(name, help=help_text, add_help=False)
        p.add_argument("rest", nargs=argparse.REMAINDER)
        p.set_defaults(func=func)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
