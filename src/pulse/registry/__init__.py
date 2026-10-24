"""Model registry: signed releases and staged rollout across a device fleet.

A device must never load a model the publisher did not sign, and a new model
must never reach the whole fleet in one step. ``signing`` provides Ed25519
keys, ``releases`` the signed manifest format and its verification, and
``rollout`` the canary, pilot, fleet ladder with deterministic device
bucketing and a health guard that blocks promotion.
"""

from pulse.registry.releases import ReleaseManifest, VerificationResult, sign_release, verify_release
from pulse.registry.rollout import (
    CanaryGuard,
    CohortStats,
    ModelRegistry,
    RolloutState,
    Stage,
    device_bucket,
)
from pulse.registry.signing import SigningKey, VerifyKey, sha256_hex

__all__ = [
    "CanaryGuard",
    "CohortStats",
    "ModelRegistry",
    "ReleaseManifest",
    "RolloutState",
    "SigningKey",
    "Stage",
    "VerificationResult",
    "VerifyKey",
    "device_bucket",
    "sha256_hex",
    "sign_release",
    "verify_release",
]
