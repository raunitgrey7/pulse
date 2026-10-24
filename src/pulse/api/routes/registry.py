"""Model registry: publish signed releases, walk the rollout ladder, assign devices."""

from __future__ import annotations

import base64
import binascii
from dataclasses import asdict

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from pulse.api.auth import require_api_key
from pulse.api.schemas import AssignmentOut, ReleaseIn, ReleaseOut, RollbackIn, RolloutOut
from pulse.api.service import Services
from pulse.registry.releases import ReleaseManifest, verify_release
from pulse.registry.rollout import RolloutState, Stage, device_bucket
from pulse.registry.signing import VerifyKey

router = APIRouter(prefix="/v1/registry", dependencies=[Depends(require_api_key)])


def services(request: Request) -> Services:
    return request.app.state.services


def _rollout_out(state: RolloutState) -> RolloutOut:
    return RolloutOut(version=state.version, stage=state.stage.value, percent=state.percent, history=state.history)


def _release_out(svc: Services, version: str, with_cohort: bool = False) -> ReleaseOut:
    cohort = asdict(svc.store.cohort_stats(version)) if with_cohort else None
    return ReleaseOut(
        manifest=asdict(svc.registry.manifest(version)),
        rollout=_rollout_out(svc.registry.state(version)),
        cohort=cohort,
    )


def _require_version(svc: Services, version: str) -> None:
    if version not in svc.registry.versions():
        raise HTTPException(status_code=404, detail="unknown release")


@router.get("/releases", response_model=list[ReleaseOut])
def list_releases(svc: Services = Depends(services)) -> list[ReleaseOut]:
    return [_release_out(svc, v, with_cohort=True) for v in svc.registry.versions()]


@router.post("/releases", response_model=ReleaseOut, status_code=201)
def publish(body: ReleaseIn, svc: Services = Depends(services)) -> ReleaseOut:
    """Verify the signature with the supplied key (and the trusted key when configured), then store."""
    try:
        manifest = ReleaseManifest.from_dict(body.manifest)  # type: ignore[arg-type]
        model_bytes = base64.b64decode(body.model_b64, validate=True)
    except (TypeError, ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=422, detail=f"malformed release: {exc}") from exc
    key = VerifyKey.from_hex(body.signer_public_key_hex)
    if svc.trusted_key is not None and key.fingerprint != svc.trusted_key.fingerprint:
        raise HTTPException(status_code=403, detail="release not signed by the trusted key")
    result = verify_release(key, manifest, model_bytes)
    if not result.ok:
        raise HTTPException(status_code=403, detail=f"release rejected: {result.reason}")
    try:
        svc.registry.publish(manifest, model_bytes)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    svc.refresh_gauges()
    return _release_out(svc, manifest.version)


@router.get("/releases/{version}", response_model=ReleaseOut)
def get_release(version: str, svc: Services = Depends(services)) -> ReleaseOut:
    _require_version(svc, version)
    return _release_out(svc, version, with_cohort=True)


@router.get("/releases/{version}/model")
def get_model(version: str, svc: Services = Depends(services)) -> Response:
    _require_version(svc, version)
    return Response(content=svc.registry.model_bytes(version), media_type="application/octet-stream")


@router.post("/releases/{version}/promote", response_model=ReleaseOut)
def promote(version: str, svc: Services = Depends(services)) -> ReleaseOut:
    _require_version(svc, version)
    incumbent_version = svc.registry.incumbent(exclude=version)
    incumbent = svc.store.cohort_stats(incumbent_version) if incumbent_version else None
    try:
        svc.registry.promote(version, svc.guard, svc.store.cohort_stats(version), incumbent)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    svc.refresh_gauges()
    return _release_out(svc, version, with_cohort=True)


@router.post("/releases/{version}/rollback", response_model=ReleaseOut)
def rollback(version: str, body: RollbackIn, svc: Services = Depends(services)) -> ReleaseOut:
    _require_version(svc, version)
    svc.registry.rollback(version, body.reason)
    svc.refresh_gauges()
    return _release_out(svc, version)


@router.get("/assignment/{device_id}", response_model=AssignmentOut)
def assignment(device_id: str, svc: Services = Depends(services)) -> AssignmentOut:
    version = svc.registry.assignment(device_id)
    manifest = asdict(svc.registry.manifest(version)) if version else None
    return AssignmentOut(
        device_id=device_id,
        bucket=device_bucket(device_id, svc.registry.salt),
        version=version,
        manifest=manifest,
        model_path=f"/v1/registry/releases/{version}/model" if version else None,
    )


__all__ = ["Stage", "router"]
