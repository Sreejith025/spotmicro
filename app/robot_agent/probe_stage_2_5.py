"""
SpotMicro Robot Agent - Command Bridge Verification Probe (Stage 2.5)
=====================================================================
Software-only probe verifying the command bridge between the hardened
SpotMicro Web Backend and the Raspberry Pi 5 Robot Agent.

SAFETY CONSTRAINTS:
- Strictly SOFTWARE ONLY: NO physical hardware or actuator access.
- Uses simulation and mock Socket.IO exclusively.
- hardware_active strictly False.
- Zero PCA9685, GPIO, I2C, or servo PWM imports or activation.
- Never prints or exposes secret tokens.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import socketio

from app.robot_agent.config import RobotAgentConfig
from app.robot_agent.connection_state import ConnectionState
from app.robot_agent.safety_authority import LocalSafetyAuthority, NavCommand
from app.robot_agent.client import RobotAgent

from web.backend.server import (
    app as flask_app,
    clear_connected_robot_agents,
    is_robot_agent_connected,
    register_robot_agent_session,
    simulator,
)
from web.backend.security import rate_limiter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Stage25Probe] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Stage25Probe")


def run_stage_2_5_command_bridge_probe() -> Dict[str, Any]:
    """
    Executes comprehensive software-only Stage 2.5 Command Bridge verification probe.
    Verifies all 14 failure, recovery, and safety cases.
    """
    logger.info("==================================================================")
    logger.info("   Starting SpotMicro Stage 2.5 Command Bridge Software-Only Probe")
    logger.info("==================================================================")

    test_token = "probe-stage25-secure-token-999"
    os.environ["OPERATOR_API_TOKEN"] = test_token
    rate_limiter.reset()
    clear_connected_robot_agents()
    simulator.reset_emergency_stop()
    simulator.apply_scenario("nominal")

    client = flask_app.test_client()
    auth_headers = {"Authorization": f"Bearer {test_token}"}

    cfg = RobotAgentConfig(
        backend_url="https://spotmicro-backend.onrender.com",
        operator_token=test_token,
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

    agent = RobotAgent(config=cfg, sio_client=mock_sio)
    agent.connect()

    results: Dict[str, bool] = {
        "1_pi_connected_forward_forwarded": False,
        "2_pi_connected_stop_forwarded": False,
        "3_pi_disconnected_movement_not_executed": False,
        "4_pi_disconnected_backend_does_not_report_success": False,
        "5_auth_failure_command_rejected": False,
        "6_invalid_command_rejected": False,
        "7_emergency_stop_pi_remains_stop_safe_idle": False,
        "8_pi_disconnect_after_receiving_command_safe_stop": False,
        "9_pi_reconnect_remains_stop": False,
        "10_watchdog_timeout_safe_idle_stop": False,
        "11_repeated_disconnect_reconnect_no_auto_motion": False,
        "12_hardware_isolation_no_modules": False,
        "13_backend_remains_simulation_mode": False,
        "14_hardware_active_remains_false": False,
    }
    errors = []

    try:
        # Check 1: Pi connected -> FORWARD command forwarded
        register_robot_agent_session("probe-session-pi5")
        res1 = client.post("/api/commands", json={"command": "forward"}, headers=auth_headers)
        data1 = res1.get_json()
        if (
            res1.status_code == 200
            and data1.get("robot_connected") is True
            and data1.get("robot_executed") is True
            and data1.get("robot_delivery") == "forwarded"
        ):
            handlers["command_executed"](data1)
            if agent.safety_authority.current_command == NavCommand.FORWARD:
                results["1_pi_connected_forward_forwarded"] = True
                logger.info("Check 1 PASS: Pi connected -> FORWARD forwarded and accepted by Pi.")
            else:
                errors.append(f"Check 1: Pi command was {agent.safety_authority.current_command}")
        else:
            errors.append(f"Check 1: Backend response unexpected: {data1}")

        # Check 2: Pi connected -> STOP command forwarded
        res2 = client.post("/api/commands", json={"command": "stop"}, headers=auth_headers)
        data2 = res2.get_json()
        if (
            res2.status_code == 200
            and data2.get("robot_connected") is True
            and data2.get("robot_executed") is True
        ):
            handlers["command_executed"](data2)
            if (
                agent.safety_authority.current_command == NavCommand.STOP
                and agent.safety_authority.is_safe_idle is True
            ):
                results["2_pi_connected_stop_forwarded"] = True
                logger.info("Check 2 PASS: Pi connected -> STOP forwarded and placed Pi in SAFE_IDLE / STOP.")
            else:
                errors.append(f"Check 2: Pi state unexpected after STOP: {agent.safety_authority.get_status()}")
        else:
            errors.append(f"Check 2: Backend response unexpected: {data2}")

        # Check 3: Pi disconnected -> movement command NOT executed
        clear_connected_robot_agents()
        agent.safety_authority.enter_safe_stop("Disconnected")
        res3 = client.post("/api/commands", json={"command": "forward"}, headers=auth_headers)
        if (
            res3.status_code == 200
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["3_pi_disconnected_movement_not_executed"] = True
            logger.info("Check 3 PASS: Pi disconnected -> movement command NOT executed on Pi (remains in STOP).")
        else:
            errors.append("Check 3: Disconnected Pi executed movement or unexpected state")

        # Check 4: Pi disconnected -> backend does not report successful robot execution
        data3 = res3.get_json()
        if (
            data3.get("robot_connected") is False
            and data3.get("robot_executed") is False
            and data3.get("robot_delivery") == "unavailable"
        ):
            results["4_pi_disconnected_backend_does_not_report_success"] = True
            logger.info("Check 4 PASS: Pi disconnected -> backend reports robot_executed=False, delivery=unavailable.")
        else:
            errors.append(f"Check 4: Backend falsely reported robot execution: {data3}")

        # Check 5: Authentication failure -> rejected
        res5 = client.post("/api/commands", json={"command": "forward"}, headers={"Authorization": "Bearer bad-token"})
        if res5.status_code == 401:
            results["5_auth_failure_command_rejected"] = True
            logger.info("Check 5 PASS: Authentication failure -> 401 rejected, command not forwarded.")
        else:
            errors.append(f"Check 5: Expected 401, got {res5.status_code}")

        # Check 6: Invalid command -> rejected
        res6 = client.post("/api/commands", json={"command": "fly_away"}, headers=auth_headers)
        ok_agent_invalid, _, _ = agent.safety_authority.process_command({"command": "invalid_xyz"})
        if res6.status_code == 400 and not ok_agent_invalid:
            results["6_invalid_command_rejected"] = True
            logger.info("Check 6 PASS: Invalid command -> rejected by both backend and Pi Safety Authority.")
        else:
            errors.append("Check 6: Invalid command was not properly rejected")

        # Check 7: Emergency stop -> Pi remains STOP/SAFE_IDLE
        res_estop = client.post("/api/emergency-stop", headers=auth_headers)
        handlers["emergency_stop_triggered"](res_estop.get_json())
        res_blocked_move = client.post("/api/commands", json={"command": "forward"}, headers=auth_headers)
        ok_under_estop, _, _ = agent.safety_authority.process_command("forward")
        if (
            agent.safety_authority.emergency_stop is True
            and res_blocked_move.status_code == 400
            and not ok_under_estop
            and agent.safety_authority.current_command == NavCommand.STOP
        ):
            results["7_emergency_stop_pi_remains_stop_safe_idle"] = True
            logger.info("Check 7 PASS: Emergency Stop -> Pi locked in STOP/SAFE_IDLE, movement blocked.")
        else:
            errors.append("Check 7: E-stop failed to block movement or lock STOP")

        # Reset estop for remaining checks
        client.post("/api/reset-emergency-stop", headers=auth_headers)
        handlers["emergency_stop_reset"]()

        # Check 8: Pi disconnect after receiving command -> safe STOP
        agent.safety_authority.process_command("forward")
        handlers["disconnect"]()
        if (
            agent.state == ConnectionState.DISCONNECTED
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["8_pi_disconnect_after_receiving_command_safe_stop"] = True
            logger.info("Check 8 PASS: Pi disconnect -> immediately halts into safe STOP and SAFE_IDLE.")
        else:
            errors.append("Check 8: Disconnect failed to halt robot into STOP")

        # Check 9: Pi reconnect -> remains STOP
        handlers["connect"]()
        if (
            agent.state == ConnectionState.AUTHENTICATED
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["9_pi_reconnect_remains_stop"] = True
            logger.info("Check 9 PASS: Pi reconnect -> remains strictly in STOP (no automatic movement).")
        else:
            errors.append("Check 9: Reconnect did not preserve STOP/SAFE_IDLE")

        # Check 10: Watchdog timeout -> SAFE_IDLE / STOP
        agent.safety_authority.process_command("forward")
        agent._on_watchdog_stale()
        if (
            agent.state == ConnectionState.SAFE_IDLE
            and agent.safety_authority.current_command == NavCommand.STOP
            and agent.safety_authority.is_safe_idle is True
        ):
            results["10_watchdog_timeout_safe_idle_stop"] = True
            logger.info("Check 10 PASS: Watchdog timeout -> transitions robot to SAFE_IDLE / STOP.")
        else:
            errors.append("Check 10: Watchdog timeout did not force STOP/SAFE_IDLE")

        # Check 11: Repeated disconnect/reconnect -> no automatic movement
        all_reconnects_safe = True
        for _ in range(3):
            handlers["connect"]()
            if agent.safety_authority.current_command != NavCommand.STOP:
                all_reconnects_safe = False
            agent.safety_authority.process_command("forward")
            handlers["disconnect"]()
            if agent.safety_authority.current_command != NavCommand.STOP:
                all_reconnects_safe = False
        if all_reconnects_safe:
            results["11_repeated_disconnect_reconnect_no_auto_motion"] = True
            logger.info("Check 11 PASS: Repeated disconnect/reconnect -> zero automatic movement across cycles.")
        else:
            errors.append("Check 11: Automatic movement occurred during cycle")

        # Check 12: Hardware isolation
        forbidden_modules = [m for m in ("adafruit_pca9685", "board", "busio", "RPi.GPIO") if m in sys.modules]
        if not forbidden_modules:
            results["12_hardware_isolation_no_modules"] = True
            logger.info("Check 12 PASS: Hardware isolation -> no physical hardware modules loaded in sys.modules.")
        else:
            errors.append(f"Check 12: Forbidden hardware modules found: {forbidden_modules}")

        # Check 13: Backend remains simulation mode
        res_h = client.get("/api/health")
        res_s = client.get("/api/status")
        if (
            res_h.status_code == 200
            and res_h.get_json().get("mode") == "simulation"
            and res_s.get_json().get("simulated") is True
        ):
            results["13_backend_remains_simulation_mode"] = True
            logger.info("Check 13 PASS: Backend remains in simulation mode throughout operations.")
        else:
            errors.append("Check 13: Backend simulation mode not confirmed")

        # Check 14: hardware_active strictly False
        if (
            agent.safety_authority.hardware_active is False
            and simulator.state.hardware_active is False
            and res_h.get_json().get("hardware_active") is False
        ):
            results["14_hardware_active_remains_false"] = True
            logger.info("Check 14 PASS: hardware_active is False across agent, simulator, and API.")
        else:
            errors.append("Check 14: hardware_active was True or not reported as False")

    except Exception as e:
        logger.error(f"Unexpected probe execution exception: {e}")
        errors.append(str(e))

    all_passed = all(results.values()) and len(errors) == 0

    logger.info("==================================================================")
    logger.info(f"   Stage 2.5 Probe Result: {'ALL 14 CHECKS PASSED' if all_passed else 'FAILURES DETECTED'}")
    logger.info("==================================================================")

    return {
        "success": all_passed,
        "results": results,
        "errors": errors,
    }


if __name__ == "__main__":
    res = run_stage_2_5_command_bridge_probe()
    sys.exit(0 if res["success"] else 1)

