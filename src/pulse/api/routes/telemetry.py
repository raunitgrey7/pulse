"""Telemetry ingest, device views, alerts and the fleet summary."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from pulse.api.auth import require_api_key
from pulse.api.schemas import (
    AlertOut,
    DeviceDetail,
    DeviceSummary,
    FleetSummary,
    HealthScoreOut,
    IngestResponse,
    TelemetryBatch,
    TelemetryPointOut,
)
from pulse.api.service import Services
from pulse.api.store import DeviceRow
from pulse.health.scoring import HealthScore

router = APIRouter(prefix="/v1", dependencies=[Depends(require_api_key)])


def services(request: Request) -> Services:
    return request.app.state.services


def _score_out(score: HealthScore | None) -> HealthScoreOut | None:
    if score is None:
        return None
    return HealthScoreOut(
        score=score.score,
        band=score.band,
        zone=score.zone,
        dominant_class=score.dominant_class,
        components=score.components,
    )


def _summary(svc: Services, device: DeviceRow) -> DeviceSummary:
    scores = svc.store.recent_scores(device.device_id, 30)
    rows = svc.store.recent_rows(device.device_id, 1)
    last = rows[-1] if rows else None
    return DeviceSummary(
        device_id=device.device_id,
        first_seen=device.first_seen,
        last_seen=device.last_seen,
        model_version=device.model_version,
        n_reports=device.n_reports,
        baseline_rms_g=device.baseline_rms,
        health=_score_out(scores[-1] if scores else None),
        recent_scores=[s.score for s in scores],
        last_rpm=float(last["rpm"]) if last else None,
        last_rms_g=float(last["rms_g"]) if last else None,
    )


@router.post("/telemetry", response_model=IngestResponse, status_code=202)
def ingest(batch: TelemetryBatch, svc: Services = Depends(services)) -> IngestResponse:
    outcome = svc.ingest(batch.reports)
    return IngestResponse(
        accepted=outcome.accepted, devices_updated=len(outcome.devices), alerts_opened=outcome.alerts_opened
    )


@router.get("/devices", response_model=list[DeviceSummary])
def list_devices(svc: Services = Depends(services)) -> list[DeviceSummary]:
    return [_summary(svc, d) for d in svc.store.devices()]


@router.get("/devices/{device_id}", response_model=DeviceDetail)
def device_detail(
    device_id: str, limit: int = Query(default=50, ge=1, le=500), svc: Services = Depends(services)
) -> DeviceDetail:
    device = svc.store.device(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="unknown device")
    recent = [
        TelemetryPointOut(**{k: r[k] for k in TelemetryPointOut.model_fields})
        for r in svc.store.recent_rows(device_id, limit)
    ]
    return DeviceDetail(**_summary(svc, device).model_dump(), recent=recent)


@router.get("/alerts", response_model=list[AlertOut])
def list_alerts(
    open: bool = Query(default=False), device_id: str | None = None, svc: Services = Depends(services)
) -> list[AlertOut]:
    return [AlertOut(**a.to_dict()) for a in svc.alerts.list(open_only=open, device_id=device_id)]


@router.post("/alerts/{alert_id}/ack", response_model=AlertOut)
def acknowledge(alert_id: int, svc: Services = Depends(services)) -> AlertOut:
    if svc.alerts.get(alert_id) is None:
        raise HTTPException(status_code=404, detail="unknown alert")
    return AlertOut(**svc.alerts.acknowledge(alert_id).to_dict())


@router.post("/alerts/check-stale", response_model=dict[str, int])
def check_stale(svc: Services = Depends(services)) -> dict[str, int]:
    return {"opened": svc.check_staleness()}


@router.get("/fleet/summary", response_model=FleetSummary)
def fleet_summary(svc: Services = Depends(services)) -> FleetSummary:
    latest = svc.store.latest_scores()
    bands = {"healthy": 0, "watch": 0, "critical": 0}
    for score in latest.values():
        bands[score.band] += 1
    mean = sum(s.score for s in latest.values()) / len(latest) if latest else None
    return FleetSummary(
        devices=len(svc.store.devices()),
        bands=bands,
        mean_score=None if mean is None else round(mean, 2),
        open_alerts=len(svc.alerts.list(open_only=True)),
        versions=svc.store.version_counts(),
        reports_total=svc.store.report_count(),
    )
