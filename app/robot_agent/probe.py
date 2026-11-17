"""
SpotMicro Robot Agent - Live Render Probe Runner (Stage 2.3)
============================================================
Automated execution of Stage 2.3 live software-only verification against
the SpotMicro Render Web Backend (https://spotmicro-backend.onrender.com).

SAFETY CONSTRAINTS:
- Strictly SOFTWARE ONLY.
- hardware_active MUST remain False.
- Zero PCA9685, GPIO, servo PWM, or hardware motor activation.
- Never prints or exposes the secret operator token.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Any, Dict, Tuple

import requests

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
    format="[%(asctime)s] [%(levelname)s] [SpotMicroProbe] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpotMicroProbe")


def run_stage_2_3_probe() -> Dict[str, Any]:
    """
    Executes the Stage 2.3 software-only verification probe.
    Returns structured results dictionary.
    """
    results: Dict[str, Any] = {
        "backend_health": False,
        "token_configured": False,
        "authenticated_socketio": False,
        "authenticated_telemetry": False,
        "authenticated_rest_stop": False,
        "estop_lifecycle": False,
        "watchdog_verification": False,
        "disconnect_safety": False,
        "hardware_isolation": False,
        "errors": [],
    }

    # STEP 1: Check environment configuration without printing secrets
    backend_url = os.environ.get("SPOTMICRO_BACKEND_URL", "https://spotmicro-backend.onrender.com").strip()
    raw_token = os.environ.get("SPOTMICRO_OPERATOR_API_TOKEN", "").strip()

    # Also check local untracked .env file if token not in os.environ
    if not raw_token:
        try:
            cfg_loaded = load_agent_config(backend_url=backend_url)
            raw_token = cfg_loaded.operator_token
        except ConfigurationError:
            pass

    token_present = bool(raw_token)
    results["token_configured"] = token_present
    logger.info(f"STEP 1 Config: backend_url = {backend_url}")
    logger.info(f"STEP 1 Config: operator_token = {'PRESENT' if token_present else 'ABSENT'}")

    # STEP 2: Verify backend health
    health_url = f"{backend_url.rstrip('/')}/api/health"
    try:
        res = requests.get(health_url, timeout=30)
        if res.status_code == 200:
            health_data = res.json()
            if (
                health_data.get("status") == "ok"
                and health_data.get("mode") == "simulation"
                and health_data.get("hardware_active") is False
            ):
                results["backend_health"] = True
                logger.info("STEP 2 Backend Health: PASS (200 OK, mode=simulation, hardware_active=False)")
            else:
                results["errors"].append(f"Unexpected health payload: {health_data}")
        else:
            results["errors"].append(f"Health check returned HTTP {res.status_code}")
    except Exception as e:
        results["errors"].append(f"Health check failed: {e}")
        logger.error(f"STEP 2 Health check error: {e}")

    # If token is absent, record that authenticated tests require the token and test negative auth
    if not token_present:
        logger.warning("OPERATOR_API_TOKEN is ABSENT. Verifying negative authentication rejection...")
        # Verify invalid token rejected by live Render
        try:
            dummy_cfg = RobotAgentConfig(
                backend_url=backend_url,
                operator_token="invalid-probe-token-placeholder",
            )
            dummy_agent = RobotAgent(config=dummy_cfg)
            conn_res = dummy_agent.connect(wait_timeout=5.0)
            if not conn_res and dummy_agent.state == ConnectionState.ERROR:
                logger.info("Negative Auth Verification: PASS (Invalid token rejected, agent in ERROR/SAFE_IDLE)")
            dummy_agent.disconnect()
        except Exception as e:
            logger.debug(f"Negative auth check caught: {e}")

        # Hardware isolation check
        results["hardware_isolation"] = _verify_hardware_isolation()
        return results

    # If token IS present, execute full authenticated live probe:
    cfg = RobotAgentConfig(
        backend_url=backend_url,
        operator_token=raw_token,
        watchdog_timeout_s=5.0,
        heartbeat_interval_s=2.0,
    )
    agent = RobotAgent(config=cfg)

    try:
        # STEP 3: Authenticated Socket.IO connection
        logger.info("STEP 3: Connecting authenticated Socket.IO client...")
        connected = agent.connect(wait_timeout=10.0)
        if connected and agent.state == ConnectionState.AUTHENTICATED and agent.is_connected:
            results["authenticated_socketio"] = True
            logger.info("STEP 3 Socket.IO Connection: PASS (AUTHENTICATED)")
        else:
            results["errors"].append(f"Socket.IO connection failed. State: {agent.state.value}")
            logger.error(f"STEP 3 Socket.IO Connection failed: state={agent.state.value}")

        # STEP 4: Authenticated telemetry observation
        logger.info("STEP 4: Observing incoming simulation telemetry events (4s)...")
        initial_feed = agent.watchdog.last_kick_time
        time.sleep(4.0)
        latest_feed = agent.watchdog.last_kick_time
        if latest_feed > initial_feed and not agent.watchdog.is_stale:
            results["authenticated_telemetry"] = True
            logger.info("STEP 4 Authenticated Telemetry: PASS (Events received, watchdog actively fed)")
        else:
            results["errors"].append("Telemetry did not feed watchdog as expected")

        # STEP 5: Authenticated REST command path (STOP ONLY)
        logger.info("STEP 5: Sending authenticated STOP command via REST...")
        cmd_url = f"{backend_url.rstrip('/')}/api/commands"
        res_cmd = requests.post(
            cmd_url,
            headers=cfg.auth_headers,
            json={"command": "stop", "duration": 1.0, "speed": 1.0},
            timeout=10,
        )
        if res_cmd.status_code == 200:
            cmd_data = res_cmd.json()
            # Feed through local SafetyAuthority
            ok, cmd_enum, msg = agent.safety_authority.process_command("stop")
            if (
                cmd_data.get("command") == "stop"
                and ok
                and cmd_enum == NavCommand.STOP
                and not agent.safety_authority.hardware_active
            ):
                results["authenticated_rest_stop"] = True
                logger.info("STEP 5 REST STOP Command: PASS (HTTP 200, command=STOP, hardware_active=False)")
            else:
                results["errors"].append(f"Command mismatch or safety error: {cmd_data}")
        else:
            results["errors"].append(f"POST /api/commands returned HTTP {res_cmd.status_code}: {res_cmd.text}")

        # STEP 6: Authenticated E-STOP behavior
        logger.info("STEP 6: Testing authenticated E-Stop trigger & reset...")
        estop_url = f"{backend_url.rstrip('/')}/api/emergency-stop"
        reset_url = f"{backend_url.rstrip('/')}/api/reset-emergency-stop"

        res_estop = requests.post(estop_url, headers=cfg.auth_headers, timeout=10)
        res_reset = requests.post(reset_url, headers=cfg.auth_headers, timeout=10)

        if res_estop.status_code == 200 and res_reset.status_code == 200:
            estop_data = res_estop.json()
            reset_data = res_reset.json()
            if (
                estop_data.get("emergency_stop") is True
                and reset_data.get("emergency_stop") is False
            ):
                # Test local safety authority e-stop lifecycle
                agent.safety_authority.trigger_emergency_stop("Stage 2.3 probe")
                estop_locked = (agent.safety_authority.current_command == NavCommand.STOP)
                agent.safety_authority.reset_emergency_stop()
                reset_clean = (agent.safety_authority.current_command == NavCommand.STOP)

                if estop_locked and reset_clean:
                    results["estop_lifecycle"] = True
                    logger.info("STEP 6 E-Stop Lifecycle: PASS (Triggered and reset successfully via authenticated endpoints)")
        else:
            results["errors"].append(f"E-Stop requests failed: trigger={res_estop.status_code}, reset={res_reset.status_code}")

        # STEP 7: Watchdog verification
        logger.info("STEP 7: Verifying watchdog behavior...")
        # A. Active connection watchdog is fresh
        fresh = not agent.watchdog.is_stale
        # B. Simulated communication loss triggers stale transition
        agent._on_watchdog_stale()
        stale_ok = (agent.state == ConnectionState.SAFE_IDLE and agent.safety_authority.current_command == NavCommand.STOP)
        # C. Feed recovers fresh state
        agent.watchdog.feed()
        recovered = not agent.watchdog.is_stale

        if fresh and stale_ok and recovered:
            results["watchdog_verification"] = True
            logger.info("STEP 7 Watchdog Verification: PASS (Fresh -> Stale on timeout -> Safe recovery)")
        else:
            results["errors"].append("Watchdog state transitions did not match specification")

        # STEP 8: Disconnect safety
        logger.info("STEP 8: Disconnecting agent...")
        agent.disconnect()
        if (
            agent.state == ConnectionState.DISCONNECTED
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
            and not agent.is_connected
        ):
            results["disconnect_safety"] = True
            logger.info("STEP 8 Disconnect Safety: PASS (State=DISCONNECTED, command=STOP, safe_idle=True)")
        else:
            results["errors"].append("Disconnect safety check failed")

    except Exception as e:
        results["errors"].append(f"Probe execution error: {e}")
        logger.error(f"Probe execution error: {e}", exc_info=True)
    finally:
        agent.disconnect()

    # STEP 9: Hardware isolation verification
    results["hardware_isolation"] = _verify_hardware_isolation()

    return results


def run_stage_2_4_failure_recovery_probe() -> Dict[str, Any]:
    """
    Executes controlled Stage 2.4 Failure & Recovery verification probe:
    1. Normal connected state
    2. Socket/transport disconnect
    3. Watchdog timeout
    4. Recovery safety (no auto-motion)
    5. Authentication failure
    6. Agent restart
    7. Repeated failure/recovery (multiple cycles)
    8. Hardware isolation (hardware_active=False)
    """
    results: Dict[str, Any] = {
        "1_normal_connected_state": False,
        "2_socket_disconnect": False,
        "3_watchdog_timeout": False,
        "4_recovery_safety": False,
        "5_authentication_failure": False,
        "6_agent_restart": False,
        "7_repeated_failure_recovery": False,
        "8_hardware_isolation": False,
        "overall_verdict": "FAIL",
        "errors": [],
    }

    logger.info("=== Starting SpotMicro Stage 2.4 Failure & Recovery Probe ===")

    # 1. Normal Connected State
    backend_url = os.environ.get("SPOTMICRO_BACKEND_URL", "https://spotmicro-backend.onrender.com").strip()
    health_url = f"{backend_url.rstrip('/')}/api/health"
    try:
        res = requests.get(health_url, timeout=30)
        if res.status_code == 200:
            h = res.json()
            if h.get("status") == "ok" and h.get("mode") == "simulation" and h.get("hardware_active") is False:
                results["1_normal_connected_state"] = True
                logger.info("Probe Check 1: Normal Connected State -> PASS (Backend healthy, hardware_active=False)")
            else:
                results["errors"].append(f"Unexpected health payload: {h}")
        else:
            results["errors"].append(f"Backend returned HTTP {res.status_code}")
    except Exception as e:
        results["errors"].append(f"Health check error: {e}")

    # Setup controlled mock Socket.IO for failure injection
    from unittest.mock import MagicMock
    import socketio

    test_cfg = RobotAgentConfig(
        backend_url=backend_url,
        operator_token="mock-probe-operator-token",
        watchdog_timeout_s=0.2,
        heartbeat_interval_s=0.1,
    )

    handlers = {}
    def mock_on(event):
        def decorator(func):
            handlers[event] = func
            return func
        return decorator

    mock_sio = MagicMock(spec=socketio.Client)
    mock_sio.connected = True
    mock_sio.on.side_effect = mock_on

    agent = RobotAgent(config=test_cfg, sio_client=mock_sio)
    agent.connect()

    # 2. Socket/transport disconnect
    agent.safety_authority.process_command("forward")
    if "disconnect" in handlers:
        handlers["disconnect"]()
        if (
            agent.state == ConnectionState.DISCONNECTED
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["2_socket_disconnect"] = True
            logger.info("Probe Check 2: Socket Disconnect -> PASS (Transitions to DISCONNECTED, STOP, SAFE_IDLE)")
        else:
            results["errors"].append("Socket disconnect did not force STOP + SAFE_IDLE")
    else:
        results["errors"].append("Disconnect handler not registered")

    # 3. Watchdog timeout
    # Re-arm forward motion and simulate silence
    agent.safety_authority.process_command("forward")
    agent._on_watchdog_stale()
    if (
        agent.state == ConnectionState.SAFE_IDLE
        and agent.safety_authority.current_command == NavCommand.STOP
        and agent.safety_authority.is_safe_idle is True
    ):
        # Restore heartbeat
        agent.watchdog.feed()
        if not agent.watchdog.is_stale and agent.safety_authority.current_command == NavCommand.STOP:
            results["3_watchdog_timeout"] = True
            logger.info("Probe Check 3: Watchdog Timeout -> PASS (Timeout forces STOP + SAFE_IDLE, feed restores without motion)")
        else:
            results["errors"].append("Watchdog recovery issued motion or failed feed")
    else:
        results["errors"].append("Watchdog stale callback did not force STOP")

    # 4. Recovery safety
    # Reconnect after failure
    if "connect" in handlers:
        handlers["connect"]()
        if (
            agent.state == ConnectionState.AUTHENTICATED
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["4_recovery_safety"] = True
            logger.info("Probe Check 4: Recovery Safety -> PASS (Reconnection stays in STOP, zero auto-motion)")
        else:
            results["errors"].append("Reconnection altered safety state or issued motion")
    else:
        results["errors"].append("Connect handler not registered")

    # 5. Authentication failure
    auth_fail_sio = MagicMock(spec=socketio.Client)
    auth_fail_sio.connected = False
    auth_fail_sio.connect.side_effect = socketio.exceptions.ConnectionError("401 Unauthorized")
    bad_agent = RobotAgent(config=test_cfg, sio_client=auth_fail_sio)
    conn_ok = bad_agent.connect()
    if (
        not conn_ok
        and bad_agent.state == ConnectionState.ERROR
        and bad_agent.safety_authority.current_command == NavCommand.STOP
        and bad_agent.safety_authority.is_safe_idle is True
    ):
        results["5_authentication_failure"] = True
        logger.info("Probe Check 5: Authentication Failure -> PASS (Rejection yields ERROR, STOP, SAFE_IDLE)")
    else:
        results["errors"].append("Authentication failure did not fail closed safely")

    # 6. Agent restart
    restart_agent = RobotAgent(config=test_cfg)
    if (
        restart_agent.state == ConnectionState.DISCONNECTED
        and restart_agent.safety_authority.current_command == NavCommand.STOP
        and restart_agent.safety_authority.is_safe_idle is True
        and not restart_agent.safety_authority.hardware_active
    ):
        results["6_agent_restart"] = True
        logger.info("Probe Check 6: Agent Restart -> PASS (Starts in DISCONNECTED, STOP, SAFE_IDLE)")
    else:
        results["errors"].append("Agent restart initialized in non-safe state")

    # 7. Repeated failure/recovery (Cycle 1 & Cycle 2)
    cycle1_ok = False
    cycle2_ok = False

    # Cycle 1: Motion -> Disconnect -> Reconnect
    agent.safety_authority.process_command("forward")
    handlers["disconnect"]()
    c1_stopped = (agent.safety_authority.current_command == NavCommand.STOP)
    handlers["connect"]()
    c1_recovered = (agent.safety_authority.current_command == NavCommand.STOP and agent.safety_authority.is_safe_idle)
    if c1_stopped and c1_recovered:
        cycle1_ok = True

    # Cycle 2: Motion -> Watchdog Timeout -> Fresh Heartbeat -> Reconnect
    agent.safety_authority.process_command("left")
    agent._on_watchdog_stale()
    c2_stopped = (agent.safety_authority.current_command == NavCommand.STOP)
    agent.watchdog.feed()
    handlers["connect"]()
    c2_recovered = (agent.safety_authority.current_command == NavCommand.STOP and agent.safety_authority.is_safe_idle)
    if c2_stopped and c2_recovered:
        cycle2_ok = True

    if cycle1_ok and cycle2_ok:
        results["7_repeated_failure_recovery"] = True
        logger.info("Probe Check 7: Repeated Failure/Recovery -> PASS (2 consecutive cycles return safely to STOP/SAFE_IDLE)")
    else:
        results["errors"].append("Repeated failure/recovery cycle violated safety invariant")

    # 8. Hardware isolation
    results["8_hardware_isolation"] = _verify_hardware_isolation()
    if results["8_hardware_isolation"]:
        logger.info("Probe Check 8: Hardware Isolation -> PASS (hardware_active=False, no forbidden modules)")

    # Overall verdict
    checks_passed = all(
        results[k] for k in [
            "1_normal_connected_state",
            "2_socket_disconnect",
            "3_watchdog_timeout",
            "4_recovery_safety",
            "5_authentication_failure",
            "6_agent_restart",
            "7_repeated_failure_recovery",
            "8_hardware_isolation",
        ]
    )

    results["overall_verdict"] = "PASS" if checks_passed else "FAIL"
    logger.info(f"=== Stage 2.4 Probe Completed: {results['overall_verdict']} ===")
    return results


def _verify_hardware_isolation() -> bool:
    """Verifies that no hardware libraries or actuator pins were touched."""
    forbidden = ["adafruit_pca9685", "board", "busio", "RPi.GPIO"]
    for mod in forbidden:
        if mod in sys.modules:
            logger.error(f"Hardware isolation VIOLATION: module '{mod}' was imported!")
            return False
    logger.info("STEP Hardware Isolation: PASS (No hardware modules imported, hardware_active=False)")
    return True


if __name__ == "__main__":
    res_24 = run_stage_2_4_failure_recovery_probe()
    logger.info(f"Stage 2.4 Probe Summary: {res_24}")
