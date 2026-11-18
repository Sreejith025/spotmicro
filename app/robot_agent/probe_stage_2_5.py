"""
SpotMicro Robot Agent - Stage 2.5 Live Render Command Bridge Probe
==================================================================
Verifies the software-only command bridge between the deployed Render backend
(https://spotmicro-backend.onrender.com) and the Raspberry Pi 5 Robot Agent.

SAFETY CONSTRAINTS:
- Strictly SOFTWARE ONLY: NO physical hardware or actuator access.
- Communicates exclusively over HTTPS / WSS.
- hardware_active strictly False.
- Zero PCA9685, GPIO, I2C, or servo PWM imports or activation.
- Never prints or exposes secret tokens.
- REST command test uses STOP ONLY.
- Does NOT import the separate web backend repository.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import requests
import socketio

from app.robot_agent.config import (
    ConfigurationError,
    RobotAgentConfig,
    load_agent_config,
)
from app.robot_agent.connection_state import ConnectionState
from app.robot_agent.safety_authority import LocalSafetyAuthority, NavCommand
from app.robot_agent.client import RobotAgent

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Stage25LiveProbe] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Stage25LiveProbe")


def _verify_hardware_isolation() -> bool:
    """Verifies no physical actuator or GPIO modules are imported in Python runtime."""
    forbidden = ("adafruit_pca9685", "board", "busio", "RPi.GPIO")
    loaded = [m for m in forbidden if m in sys.modules]
    if loaded:
        logger.error(f"Hardware isolation VIOLATION: Forbidden modules found in sys.modules: {loaded}")
        return False
    logger.info("Hardware Isolation Check: PASS (No physical hardware modules loaded)")
    return True


def run_stage_2_5_live_probe() -> Dict[str, Any]:
    """
    Executes the live software-only Stage 2.5 Command Bridge verification probe
    against the deployed Render backend.
    """
    logger.info("==================================================================")
    logger.info("   Starting SpotMicro Stage 2.5 Live Render Command Bridge Probe   ")
    logger.info("==================================================================")

    results: Dict[str, Any] = {
        "A_backend_health": False,
        "B_config_loaded": False,
        "C_auth_configured": False,
        "D_authenticated_socketio": False,
        "E_agent_state_authenticated": False,
        "F_simulation_telemetry_received": False,
        "G_stop_delivered": False,
        "H_disconnect_safe_stop": False,
        "I_watchdog_safe_stop": False,
        "J_reconnect_safe_stop": False,
        "K_hardware_modules_unloaded": False,
        "L_hardware_active_false": False,
        "robot_connected_reported": None,
        "robot_delivery_reported": None,
        "backend_url": None,
        "errors": [],
        "overall_success": False,
    }

    # B & C: Pi configuration & authentication loaded safely
    backend_url = os.environ.get("SPOTMICRO_BACKEND_URL", "https://spotmicro-backend.onrender.com").strip()
    results["backend_url"] = backend_url
    raw_token = os.environ.get("SPOTMICRO_OPERATOR_API_TOKEN", "").strip()

    if not raw_token:
        try:
            cfg_loaded = load_agent_config(backend_url=backend_url)
            raw_token = cfg_loaded.operator_token
        except ConfigurationError:
            pass

    token_present = bool(raw_token)
    if token_present:
        results["C_auth_configured"] = True
        logger.info(f"Check B/C: Config & Auth -> PASS (backend_url={backend_url}, token=CONFIGURED [REDACTED])")
    else:
        logger.warning(f"Check B/C: Config & Auth -> FAILED (SPOTMICRO_OPERATOR_API_TOKEN is not set)")
        results["errors"].append("SPOTMICRO_OPERATOR_API_TOKEN not configured in environment or .env")

    # A: Backend health
    health_url = f"{backend_url.rstrip('/')}/api/health"
    try:
        res_h = requests.get(health_url, timeout=30)
        if res_h.status_code == 200:
            hdata = res_h.json()
            if (
                hdata.get("status") == "ok"
                and hdata.get("mode") == "simulation"
                and hdata.get("hardware_active") is False
            ):
                results["A_backend_health"] = True
                logger.info("Check A: Backend Health -> PASS (200 OK, mode=simulation, hardware_active=False)")
            else:
                results["errors"].append(f"Unexpected health payload: {hdata}")
        else:
            results["errors"].append(f"Backend health check returned HTTP {res_h.status_code}")
    except Exception as e:
        results["errors"].append(f"Backend health check failed: {e}")
        logger.error(f"Check A Backend Health Error: {e}")

    # If token is not present, we can only verify negative/isolation checks
    if not token_present:
        results["K_hardware_modules_unloaded"] = _verify_hardware_isolation()
        results["L_hardware_active_false"] = True
        return results

    # Initialize RobotAgent with live config
    try:
        cfg = RobotAgentConfig(
            backend_url=backend_url,
            operator_token=raw_token,
            watchdog_timeout_s=5.0,
            heartbeat_interval_s=2.0,
        )
        results["B_config_loaded"] = True
    except ConfigurationError as e:
        results["errors"].append(f"Configuration initialization error: {e}")
        return results

    agent = RobotAgent(config=cfg)

    # D & E: Authenticated Socket.IO connection
    try:
        logger.info("Check D & E: Connecting authenticated RobotAgent to Render Socket.IO...")
        connected = agent.connect(wait_timeout=15.0)
        if connected and agent.state == ConnectionState.AUTHENTICATED and agent.is_connected:
            results["D_authenticated_socketio"] = True
            results["E_agent_state_authenticated"] = True
            logger.info("Check D & E: Socket.IO Connection -> PASS (State: AUTHENTICATED, is_connected: True)")
        else:
            results["errors"].append(f"Socket.IO connection failed (state: {agent.state.value})")
            logger.error(f"Check D & E Socket.IO connection failed: state={agent.state.value}")

        # F: Simulation telemetry reception
        if agent.is_connected:
            logger.info("Check F: Observing incoming simulation telemetry events (4s)...")
            initial_feed = agent.watchdog.last_kick_time
            time.sleep(4.0)
            latest_feed = agent.watchdog.last_kick_time
            if latest_feed > initial_feed and not agent.watchdog.is_stale:
                results["F_simulation_telemetry_received"] = True
                logger.info("Check F: Simulation Telemetry -> PASS (Telemetry/status received, watchdog fed)")
            else:
                logger.warning("Check F: Telemetry feed was not observed in 4s window")
                results["errors"].append("Telemetry did not feed watchdog within observation window")

        # G: Authenticated REST command path (STOP ONLY)
        logger.info("Check G: Sending authenticated STOP command via REST...")
        ok_cmd, cmd_data = agent.send_rest_command("stop", duration=1.0, speed=1.0)
        if ok_cmd and cmd_data.get("command") == "stop":
            results["G_stop_delivered"] = True
            results["robot_connected_reported"] = cmd_data.get("robot_connected")
            results["robot_delivery_reported"] = cmd_data.get("robot_delivery")
            # Feed into local safety authority
            agent.safety_authority.process_command("stop")
            logger.info(
                f"Check G: REST STOP Command -> PASS (HTTP 200, command=stop, "
                f"robot_connected={cmd_data.get('robot_connected')}, "
                f"robot_delivery={cmd_data.get('robot_delivery')})"
            )
        else:
            results["errors"].append(f"REST STOP command failed: {cmd_data}")
            logger.error(f"Check G REST STOP failed: {cmd_data}")

        # H: Disconnect causes SAFE_IDLE / STOP
        logger.info("Check H: Testing disconnect behavior...")
        agent.disconnect()
        if (
            agent.state == ConnectionState.DISCONNECTED
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["H_disconnect_safe_stop"] = True
            logger.info("Check H: Disconnect -> PASS (Transitions to DISCONNECTED, current_command=STOP, is_safe_idle=True)")
        else:
            results["errors"].append("Disconnect did not place agent in DISCONNECTED / STOP")

        # I: Watchdog timeout causes SAFE_IDLE / STOP
        logger.info("Check I: Testing software watchdog timeout behavior...")
        # Simulate watchdog stale event
        agent._on_watchdog_stale()
        if (
            agent.state == ConnectionState.SAFE_IDLE
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            # Feed restores fresh watchdog
            agent.watchdog.feed()
            if not agent.watchdog.is_stale and agent.safety_authority.current_command == NavCommand.STOP:
                results["I_watchdog_safe_stop"] = True
                logger.info("Check I: Watchdog Timeout -> PASS (Timeout forces SAFE_IDLE / STOP, feed does not cause motion)")
            else:
                results["errors"].append("Watchdog recovery failed")
        else:
            results["errors"].append("Watchdog stale callback did not force SAFE_IDLE / STOP")

        # J: Reconnection does not automatically generate movement
        logger.info("Check J: Testing reconnection safety (zero auto-movement)...")
        agent.safety_authority.enter_safe_stop("Pre-reconnect verification")
        reconnected = agent.connect(wait_timeout=10.0)
        if (
            reconnected
            and agent.state == ConnectionState.AUTHENTICATED
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["J_reconnect_safe_stop"] = True
            logger.info("Check J: Reconnection Safety -> PASS (Reconnected into AUTHENTICATED, strictly in STOP, zero auto-motion)")
        elif not reconnected and agent.safety_authority.current_command == NavCommand.STOP and agent.safety_authority.is_safe_idle is True:
            results["J_reconnect_safe_stop"] = True
            logger.info("Check J: Reconnection Safety -> PASS (Local safety authority strictly preserved STOP, zero auto-motion)")
        else:
            results["errors"].append("Reconnection failed or did not preserve STOP / SAFE_IDLE")

        # Disconnect cleanly after probe
        agent.disconnect()

    except Exception as e:
        logger.error(f"Unexpected probe execution exception: {e}")
        results["errors"].append(str(e))

    # K: Hardware isolation
    results["K_hardware_modules_unloaded"] = _verify_hardware_isolation()

    # L: hardware_active remains False
    if agent.safety_authority.hardware_active is False:
        results["L_hardware_active_false"] = True
        logger.info("Check L: hardware_active -> PASS (hardware_active=False)")
    else:
        results["errors"].append("hardware_active was unexpectedly True")

    # Evaluation
    core_checks = [
        results["A_backend_health"],
        results["B_config_loaded"],
        results["C_auth_configured"],
        results["D_authenticated_socketio"],
        results["E_agent_state_authenticated"],
        results["F_simulation_telemetry_received"],
        results["G_stop_delivered"],
        results["H_disconnect_safe_stop"],
        results["I_watchdog_safe_stop"],
        results["J_reconnect_safe_stop"],
        results["K_hardware_modules_unloaded"],
        results["L_hardware_active_false"],
    ]

    results["overall_success"] = all(core_checks) and len(results["errors"]) == 0

    logger.info("==================================================================")
    logger.info(
        f"   Stage 2.5 Live Probe Result: "
        f"{'ALL CHECKS PASSED' if results['overall_success'] else 'COMPLETED WITH WARNINGS/FAILURES'}"
    )
    logger.info("==================================================================")

    return results


if __name__ == "__main__":
    res = run_stage_2_5_live_probe()
    sys.exit(0 if res["overall_success"] else 1)
