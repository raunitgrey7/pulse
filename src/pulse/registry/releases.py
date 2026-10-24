"""Signed release manifests.

A release is a model artefact plus a manifest. The manifest records the
artefact's SHA-256, size, kind and benchmark metrics, and the signature is
computed over the canonical JSON of the manifest fields. A device therefore
checks two things before loading a model: the signature proves the manifest
came from the publisher, and the hash proves the bytes match the manifest.
Tampering with either one fails verification.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from pulse.registry.signing import SigningKey, VerifyKey, sha256_hex

SEMVER = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_version(version: str) -> tuple[int, int, int]:
    """Strict major.minor.patch; anything else is rejected so ordering stays total."""
    match = SEMVER.match(version)
    if not match:
        raise ValueError(f"version {version!r} is not major.minor.patch")
    major, minor, patch = (int(g) for g in match.groups())
    return major, minor, patch


@dataclass(frozen=True)
class ReleaseManifest:
    """Everything a device needs to decide whether to trust and load a model."""

    version: str
    model_name: str
    kind: str
    sha256: str
    size_bytes: int
    created_at: str
    signer_fingerprint: str
    metrics: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    signature_hex: str = ""

    def canonical_bytes(self) -> bytes:
        """The signed payload: every field except the signature, in sorted-key JSON."""
        payload = {k: v for k, v in asdict(self).items() if k != "signature_hex"}
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ReleaseManifest:
        return cls(**raw)

    @classmethod
    def from_json(cls, text: str) -> ReleaseManifest:
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str


def sign_release(
    key: SigningKey,
    version: str,
    model_name: str,
    kind: str,
    model_bytes: bytes,
    metrics: dict[str, float] | None = None,
    notes: str = "",
    created_at: str | None = None,
) -> ReleaseManifest:
    """Build and sign a manifest for ``model_bytes``."""
    parse_version(version)
    unsigned = ReleaseManifest(
        version=version,
        model_name=model_name,
        kind=kind,
        sha256=sha256_hex(model_bytes),
        size_bytes=len(model_bytes),
        created_at=created_at or datetime.now(UTC).isoformat(timespec="seconds"),
        signer_fingerprint=key.verify_key().fingerprint,
        metrics=dict(metrics or {}),
        notes=notes,
    )
    signature = key.sign(unsigned.canonical_bytes())
    return ReleaseManifest(**{**asdict(unsigned), "signature_hex": signature.hex()})


def verify_release(verify_key: VerifyKey, manifest: ReleaseManifest, model_bytes: bytes) -> VerificationResult:
    """Check signer, signature and content hash; the reason names the first failure."""
    if manifest.signer_fingerprint != verify_key.fingerprint:
        return VerificationResult(False, "manifest was signed by an unknown key")
    try:
        signature = bytes.fromhex(manifest.signature_hex)
    except ValueError:
        return VerificationResult(False, "signature is not valid hex")
    if not verify_key.verify(manifest.canonical_bytes(), signature):
        return VerificationResult(False, "signature does not match manifest")
    if sha256_hex(model_bytes) != manifest.sha256:
        return VerificationResult(False, "model bytes do not match manifest hash")
    if len(model_bytes) != manifest.size_bytes:
        return VerificationResult(False, "model size does not match manifest")
    return VerificationResult(True, "ok")
