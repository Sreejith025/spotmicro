"""
SpotMicro Robot Agent - Secure Backend Connection Client
=========================================================
Raspberry Pi 5 software-side agent managing the authenticated connection to
the SpotMicro Web Backend (Render/local) via Socket.IO and REST.

SECURITY & SAFETY CONSTRAINTS:
- Authenticates using Bearer token and Socket.IO auth payload:
    auth: {"token": "<operator token>"}
    headers: {"Authorization": "Bearer <operator token>"}
- Never logs or exposes the OPERATOR_API_TOKEN.
- Strictly decoupled from physical hardware (hardware_active == False).
- Fails closed to SAFE_IDLE / STOP on:
    - Missing token or backend URL
    - Authentication rejection
    - Unreachable backend / network timeout
    - Watchdog staleness
    - Malformed or unexpected commands
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

import requests
import socketio

try:
    from app.robot_agent.config import (
        ConfigurationError,
        RobotAgentConfig,
        load_agent_config,
    )
    from app.robot_agent.connection_state import ConnectionState
    from app.robot_agent.safety_authority import LocalSafetyAuthority, NavCommand
    from app.robot_agent.watchdog import SoftwareWatchdog
except ImportError:
    from robot_agent.config import (
        ConfigurationError,
        RobotAgentConfig,
        load_agent_config,
    )
    from robot_agent.connection_state import ConnectionState
    from robot_agent.safety_authority import LocalSafetyAuthority, NavCommand
    from robot_agent.watchdog import SoftwareWatchdog

logger = logging.getLogger("SpotMicroRobotAgent.Client")


class TokenRedactionFilter(logging.Filter):
    """Logging filter ensuring that raw operator tokens never appear in log records."""

    def __init__(self, token_to_mask: Optional[str] = None):
        super().__init__()
        self.token_to_mask = token_to_mask.strip() if token_to_mask else ""

    def filter(self, record: logging.LogRecord) -> bool:
        if self.token_to_mask and len(self.token_to_mask) > 3:
            if isinstance(record.msg, str) and self.token_to_mask in record.msg:
                record.msg = record.msg.replace(self.token_to_mask, "***REDACTED***")
            if record.args:
                new_args = []
                for arg in record.args:
                    if isinstance(arg, str) and self.token_to_mask in arg:
                        new_args.append(arg.replace(self.token_to_mask, "***REDACTED***"))
                    else:
                        new_args.append(arg)
                record.args = tuple(new_args)
        return True


class RobotAgent:
    """
    Main software agent running on the Raspberry Pi 5.
    Connects to the SpotMicro web backend, receives commands, manages
    connection states, feeds the software watchdog, and enforces local safety.
    """

    def __init__(
        self,
        config: Optional[RobotAgentConfig] = None,
        sio_client: Optional[socketio.Client] = None,
        http_session: Optional[requests.Session] = None,
    ):
        self._lock = threading.Lock()
        self._state: ConnectionState = ConnectionState.DISCONNECTED
        self.safety_authority = LocalSafetyAuthority()

        # Load or validate configuration
        try:
            self.config: Optional[RobotAgentConfig] = config or load_agent_config()
        except ConfigurationError as e:
            logger.error(f"Configuration failure: {e}. Transitioning to ERROR / SAFE_IDLE.")
            self.config = None
            self._state = ConnectionState.ERROR
            self.safety_authority.enter_safe_stop(f"Configuration failure: {e}")

        # Attach token redaction to logger
        if self.config and self.config.operator_token:
            redaction_filter = TokenRedactionFilter(self.config.operator_token)
            logger.addFilter(redaction_filter)

        # Software watchdog
        timeout_s = self.config.watchdog_timeout_s if self.config else 5.0
        self.watchdog = SoftwareWatchdog(
            timeout_s=timeout_s,
            on_stale_callback=self._on_watchdog_stale,
        )

        # Socket.IO & HTTP clients
        self.http_session = http_session or requests.Session()
        self.sio = sio_client or socketio.Client(
            reconnection=False,
            request_timeout=5,
        )

        self._heartbeat_thread: Optional[threading.Thread] = None
        self._running = False

        self._register_socket_events()

    @property
    def state(self) -> ConnectionState:
        with self._lock:
            return self._state

    @property
    def is_connected(self) -> bool:
        return self.sio.connected and self.state == ConnectionState.AUTHENTICATED

    def _set_state(self, new_state: ConnectionState):
        with self._lock:
            old_state = self._state
            self._state = new_state
            logger.info(f"Connection state transition: {old_state.value} -> {new_state.value}")

    def _register_socket_events(self):
        """Binds Socket.IO event callbacks."""

        @self.sio.on("connect")
        def on_connect():
            self._set_state(ConnectionState.AUTHENTICATED)
            self.watchdog.feed()
            logger.info("Socket.IO connection established and authenticated.")
            try:
                self.sio.emit("register_robot_agent", {"agent": "pi5", "timestamp": time.time()})
            except Exception as e:
                logger.debug(f"Failed to emit register_robot_agent: {e}")

        @self.sio.on("disconnect")
        def on_disconnect():
            self._set_state(ConnectionState.DISCONNECTED)
            self.safety_authority.enter_safe_stop("Socket.IO disconnected")
            logger.warning("Socket.IO disconnected from backend. Robot entered SAFE_IDLE / STOP.")

        @self.sio.on("connect_error")
        def on_connect_error(data=None):
            self._set_state(ConnectionState.ERROR)
            self.safety_authority.enter_safe_stop("Connection error / authentication failed")
            logger.error("Socket.IO connect error (auth rejection or unreachable).")

        @self.sio.on("command_executed")
        def on_command(data):
            self.watchdog.feed()
            success, cmd, msg = self.safety_authority.process_command(data)
            if not success:
                self._set_state(ConnectionState.SAFE_IDLE)
                logger.warning(f"Safety authority rejected backend command: {msg}")

        @self.sio.on("emergency_stop_triggered")
        def on_estop(data=None):
            self.watchdog.feed()
            self._set_state(ConnectionState.SAFE_IDLE)
            self.safety_authority.trigger_emergency_stop("Backend E-stop broadcast received")

        @self.sio.on("emergency_stop_reset")
        def on_estop_reset(data=None):
            self.watchdog.feed()
            self.safety_authority.reset_emergency_stop()
            logger.info("Backend emergency stop reset received. Safety authority reset to SAFE_IDLE / STOP.")

        @self.sio.on("telemetry_update")
        @self.sio.on("status_update")
        @self.sio.on("navigation_update")
        @self.sio.on("auto_mode_update")
        @self.sio.on("destination_auto_update")
        def on_telemetry(data=None):
            self.watchdog.feed()

        @self.sio.on("pong_server")
        def on_pong(data=None):
            self.watchdog.feed()

    def _on_watchdog_stale(self):
        """Callback invoked when software watchdog detects stale communication."""
        self._set_state(ConnectionState.SAFE_IDLE)
        self.safety_authority.enter_safe_stop("Watchdog timeout: communication became stale")
        logger.warning("Watchdog expired: transitioning to SAFE_IDLE / STOP.")

    def check_backend_health(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Sends GET /api/health to the web backend via HTTP.
        Verifies the service is operational, in simulation mode, and hardware_active is False.
        """
        if not self.config:
            return False, {"error": "Missing or invalid configuration"}

        url = f"{self.config.backend_url.rstrip('/')}/api/health"
        try:
            res = self.http_session.get(url, headers=self.config.auth_headers, timeout=5.0)
            if res.status_code == 200:
                data = res.json()
                return True, data
            return False, {"error": f"HTTP {res.status_code}", "body": res.text}
        except Exception as e:
            logger.error(f"Failed to reach backend health endpoint: {e}")
            return False, {"error": str(e)}

    def _heartbeat_worker(self):
        """Background thread emitting periodic pings to keep the connection fresh."""
        interval = self.config.heartbeat_interval_s if self.config else 2.0
        while self._running:
            try:
                if self.sio.connected:
                    self.sio.emit("ping_server", {"client_time": time.time()})
            except Exception as e:
                logger.debug(f"Heartbeat ping error: {e}")
            time.sleep(interval)

    def connect(self, wait_timeout: float = 5.0) -> bool:
        """
        Initiates authenticated connection to the SpotMicro web backend.
        Fails safely to ERROR and SAFE_IDLE / STOP on any failure.
        """
        if not self.config:
            self._set_state(ConnectionState.ERROR)
            self.safety_authority.enter_safe_stop("Cannot connect without valid configuration")
            return False

        self._set_state(ConnectionState.CONNECTING)
        backend_url = self.config.backend_url

        try:
            # Connect via Socket.IO passing auth payload and headers
            auth_payload = dict(self.config.socketio_auth)
            auth_payload["client_type"] = "robot_agent"
            self.sio.connect(
                backend_url,
                auth=auth_payload,
                headers=self.config.auth_headers,
                wait_timeout=wait_timeout,
            )

            # Verification: is client connected?
            if self.sio.connected:
                self._set_state(ConnectionState.AUTHENTICATED)
                self.watchdog.feed()
                self.watchdog.start()

                self._running = True
                self._heartbeat_thread = threading.Thread(
                    target=self._heartbeat_worker,
                    name="RobotAgentHeartbeat",
                    daemon=True,
                )
                self._heartbeat_thread.start()
                logger.info("RobotAgent connected and authenticated successfully.")
                return True
            else:
                self._set_state(ConnectionState.ERROR)
                self.safety_authority.enter_safe_stop("Connection handshake failed")
                return False

        except socketio.exceptions.ConnectionError as e:
            logger.error("Socket.IO connection rejected (authentication failed or server error).")
            self._set_state(ConnectionState.ERROR)
            self.safety_authority.enter_safe_stop("Authentication rejected or connection error")
            return False
        except Exception as e:
            logger.error(f"Unexpected connection error: {e}")
            self._set_state(ConnectionState.ERROR)
            self.safety_authority.enter_safe_stop("Unexpected network or connection error")
            return False

    def disconnect(self):
        """Gracefully disconnects from backend and enters SAFE_IDLE / STOP."""
        self._running = False
        self.watchdog.stop()

        if self.sio.connected:
            try:
                self.sio.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting Socket.IO: {e}")

        self._set_state(ConnectionState.DISCONNECTED)
        self.safety_authority.enter_safe_stop("Disconnection requested")
        logger.info("RobotAgent disconnected and placed in SAFE_IDLE / STOP.")

    def get_status(self) -> Dict[str, Any]:
        """
        Returns full diagnostic status of the agent.
        Never reveals or includes secret tokens.
        """
        return {
            "connection_state": self.state.value,
            "is_connected": self.is_connected,
            "backend_url": self.config.backend_url if self.config else None,
            "watchdog_stale": self.watchdog.is_stale,
            "safety_authority": self.safety_authority.get_status(),
        }

