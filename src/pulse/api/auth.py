"""API key authentication, enabled only when ``APP_API_KEY`` is set."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request, status


async def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Constant-time comparison against the configured key; open when none is configured."""
    expected: str | None = request.app.state.settings.api_key
    if expected is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API key")
