"""Ingest and control plane API.

FastAPI application that receives device telemetry, scores device health,
raises alerts, serves the model registry with staged rollout, exposes
Prometheus metrics and hosts the dashboard. State is SQLite for telemetry
and plain files for the registry, so a single container with one volume is
a complete deployment.
"""

from pulse.api.app import create_app

__all__ = ["create_app"]
