"""
SpotMicro Robot Agent - Configuration Module
============================================
Handles environment configuration for the Raspberry Pi robot agent.
Extracts and validates SPOTMICRO_BACKEND_URL and SPOTMICRO_OPERATOR_API_TOKEN.

SECURITY & SAFETY RULES:
- Never hardcode tokens or default secrets.
- Fails closed with ConfigurationError if token or backend URL is missing.
- Masked __repr__ and string representations prevent secret leakage in logs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse


class ConfigurationError(ValueError):
    """Raised when robot agent configuration is invalid or missing required values."""
    pass


@dataclass(frozen=True)
class RobotAgentConfig:
    """Immutable runtime configuration for SpotMicro Robot Agent."""
    backend_url: str
    operator_token: str
    watchdog_timeout_s: float = 5.0
    heartbeat_interval_s: float = 2.0

    def __post_init__(self):
        if not self.backend_url or not isinstance(self.backend_url, str):
            raise ConfigurationError("Missing or invalid SPOTMICRO_BACKEND_URL.")
        
        parsed = urlparse(self.backend_url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ConfigurationError(
                f"Invalid SPOTMICRO_BACKEND_URL '{self.backend_url}'. Must begin with http:// or https://"
            )

        if not self.operator_token or not isinstance(self.operator_token, str) or not self.operator_token.strip():
            raise ConfigurationError("Missing or empty SPOTMICRO_OPERATOR_API_TOKEN.")

        if self.watchdog_timeout_s <= 0:
            raise ConfigurationError("watchdog_timeout_s must be strictly positive.")

        if self.heartbeat_interval_s <= 0:
            raise ConfigurationError("heartbeat_interval_s must be strictly positive.")

    def __repr__(self) -> str:
        """Sanitized representation that never prints the actual secret token."""
        return (
            f"RobotAgentConfig(backend_url='{self.backend_url}', "
            f"operator_token='***REDACTED***', "
            f"watchdog_timeout_s={self.watchdog_timeout_s}, "
            f"heartbeat_interval_s={self.heartbeat_interval_s})"
        )

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def auth_headers(self) -> dict[str, str]:
        """HTTP Authorization Bearer header dictionary."""
        return {"Authorization": f"Bearer {self.operator_token.strip()}"}

    @property
    def socketio_auth(self) -> dict[str, str]:
        """Socket.IO handshake authentication dictionary."""
        return {"token": self.operator_token.strip()}


def _load_dotenv_if_present(dotenv_path: Optional[str] = None):
    """
    Safely loads environment variables from an untracked local .env file
    if present and not already defined in os.environ.
    """
    candidates = []
    if dotenv_path:
        candidates.append(dotenv_path)
    else:
        candidates.append(os.path.join(os.getcwd(), ".env"))
        # Also check project root relative to this file
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        candidates.append(os.path.join(project_root, ".env"))

    for candidate in candidates:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
                break
            except Exception:
                pass


def load_agent_config(
    backend_url: Optional[str] = None,
    operator_token: Optional[str] = None,
    watchdog_timeout_s: Optional[float] = None,
    heartbeat_interval_s: Optional[float] = None,
    dotenv_path: Optional[str] = None,
) -> RobotAgentConfig:
    """
    Loads configuration from explicitly provided parameters, local .env, or environment:
    - SPOTMICRO_BACKEND_URL
    - SPOTMICRO_OPERATOR_API_TOKEN
    - SPOTMICRO_WATCHDOG_TIMEOUT_S (optional, default 5.0)
    - SPOTMICRO_HEARTBEAT_INTERVAL_S (optional, default 2.0)

    Raises ConfigurationError if required settings are absent or invalid.
    """
    _load_dotenv_if_present(dotenv_path)

    resolved_url = backend_url or os.environ.get("SPOTMICRO_BACKEND_URL", "").strip()
    resolved_token = operator_token or os.environ.get("SPOTMICRO_OPERATOR_API_TOKEN", "").strip()

    if watchdog_timeout_s is None:
        raw_watchdog = os.environ.get("SPOTMICRO_WATCHDOG_TIMEOUT_S", "5.0").strip()
        try:
            watchdog_timeout_s = float(raw_watchdog)
        except ValueError:
            raise ConfigurationError(f"Invalid SPOTMICRO_WATCHDOG_TIMEOUT_S: {raw_watchdog}")

    if heartbeat_interval_s is None:
        raw_heartbeat = os.environ.get("SPOTMICRO_HEARTBEAT_INTERVAL_S", "2.0").strip()
        try:
            heartbeat_interval_s = float(raw_heartbeat)
        except ValueError:
            raise ConfigurationError(f"Invalid SPOTMICRO_HEARTBEAT_INTERVAL_S: {raw_heartbeat}")

    return RobotAgentConfig(
        backend_url=resolved_url,
        operator_token=resolved_token,
        watchdog_timeout_s=watchdog_timeout_s,
        heartbeat_interval_s=heartbeat_interval_s,
    )

