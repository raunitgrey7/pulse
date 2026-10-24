"""Ed25519 keys for signing releases.

Asymmetric signatures are used instead of an HMAC because the fleet only
needs the public key: a compromised device cannot forge a release. Ed25519
is chosen for its small keys (32 bytes), small signatures (64 bytes) and
deterministic signing, which keeps release manifests reproducible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

RAW = serialization.Encoding.Raw
PUBLIC_FORMAT = serialization.PublicFormat.Raw
PRIVATE_FORMAT = serialization.PrivateFormat.Raw


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class VerifyKey:
    """Public half of a signing key; this is what devices carry."""

    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != 32:
            raise ValueError("an Ed25519 public key is 32 bytes")

    @property
    def fingerprint(self) -> str:
        """Short identifier used in manifests and logs."""
        return sha256_hex(self.raw)[:16]

    def verify(self, data: bytes, signature: bytes) -> bool:
        try:
            Ed25519PublicKey.from_public_bytes(self.raw).verify(signature, data)
        except (InvalidSignature, ValueError):
            return False
        return True

    def to_hex(self) -> str:
        return self.raw.hex()

    @classmethod
    def from_hex(cls, text: str) -> VerifyKey:
        return cls(bytes.fromhex(text))


class SigningKey:
    """Private key held by the publisher only."""

    def __init__(self, private: Ed25519PrivateKey) -> None:
        self._private = private

    @classmethod
    def generate(cls) -> SigningKey:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_seed(cls, seed: bytes) -> SigningKey:
        """Deterministic key from 32 secret bytes, for tests and reproducible fixtures."""
        if len(seed) != 32:
            raise ValueError("seed must be 32 bytes")
        return cls(Ed25519PrivateKey.from_private_bytes(seed))

    def to_seed(self) -> bytes:
        return self._private.private_bytes(RAW, PRIVATE_FORMAT, serialization.NoEncryption())

    def verify_key(self) -> VerifyKey:
        return VerifyKey(self._private.public_key().public_bytes(RAW, PUBLIC_FORMAT))

    def sign(self, data: bytes) -> bytes:
        return self._private.sign(data)
