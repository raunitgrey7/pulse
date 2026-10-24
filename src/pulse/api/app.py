"""Application factory: wiring, lifespan, health, metrics and the dashboard."""

from __future__ import annotations

import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from pulse.api.logging import configure_logging, request_logging_middleware
from pulse.api.routes import registry as registry_routes
from pulse.api.routes import telemetry as telemetry_routes
from pulse.api.schemas import ServiceHealth
from pulse.api.service import Services
from pulse.api.settings import Settings

STATIC_DIR = Path(__file__).parent / "static"


def _package_version() -> str:
    try:
        return package_version("pulse")
    except PackageNotFoundError:
        return "0.0.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application; pass ``settings`` explicitly in tests."""
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = Services(settings)
        app.state.services.refresh_gauges()
        try:
            yield
        finally:
            app.state.services.close()

    app = FastAPI(title="pulse", version=_package_version(), lifespan=lifespan)
    app.state.settings = settings
    app.middleware("http")(request_logging_middleware)
    app.middleware("http")(_metrics_middleware)
    app.add_exception_handler(RequestValidationError, _validation_handler)  # type: ignore[arg-type]
    app.include_router(telemetry_routes.router)
    app.include_router(registry_routes.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/health", response_model=ServiceHealth)
    def health(request: Request) -> ServiceHealth:
        svc: Services = request.app.state.services
        return ServiceHealth(
            status="ok",
            version=app.version,
            uptime_s=round(time.time() - svc.started_at, 3),
            devices=len(svc.store.devices()),
            open_alerts=len(svc.alerts.list(open_only=True)),
            releases=len(svc.registry.versions()),
        )

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics(request: Request) -> PlainTextResponse:
        svc: Services = request.app.state.services
        return PlainTextResponse(svc.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _finite(value: Any) -> Any:
    """Replace NaN and infinities so an error echo can always be serialised."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(v) for v in value]
    return value


async def _validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """A 422 whose body echoes the offending input; the default handler crashes on NaN."""
    return JSONResponse(status_code=422, content={"detail": _finite(jsonable_encoder(exc.errors()))})


async def _metrics_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Count and time every request by route template so paths with ids do not explode."""
    start = time.perf_counter()
    response = await call_next(request)
    svc: Services | None = getattr(request.app.state, "services", None)
    if svc is not None:
        route = request.scope.get("route")
        template = getattr(route, "path", request.url.path)
        svc.metrics.http_requests.inc(method=request.method, path=template, status=str(response.status_code))
        svc.metrics.http_duration.observe(time.perf_counter() - start, method=request.method, path=template)
    return response
