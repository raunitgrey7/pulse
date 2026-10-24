"""Runtime configuration from environment variables, with safe defaults for local use."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    api_key: str | None = None
    db_path: Path = Path("data/pulse.db")
    registry_dir: Path = Path("data/registry")
    verify_key_hex: str | None = None
    alert_webhook_url: str | None = None
    stale_after_s: float = 60.0
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            api_key=env.get("APP_API_KEY") or None,
            db_path=Path(env.get("PULSE_DB_PATH", "data/pulse.db")),
            registry_dir=Path(env.get("PULSE_REGISTRY_DIR", "data/registry")),
            verify_key_hex=env.get("PULSE_VERIFY_KEY_HEX") or None,
            alert_webhook_url=env.get("PULSE_ALERT_WEBHOOK_URL") or None,
            stale_after_s=float(env.get("PULSE_STALE_AFTER_S", "60")),
            log_level=env.get("PULSE_LOG_LEVEL", "INFO"),
        )
