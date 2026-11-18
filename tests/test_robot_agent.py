"""
SpotMicro Robot Agent - Unit & Safety Test Suite (Stage 2.1)
============================================================
Comprehensive test suite verifying Raspberry Pi 5 software agent:
- Missing token fail-safe
- Missing backend URL fail-safe
- Valid configuration & secret masking
- Invalid authentication rejection
- Backend unavailable handling
- Stale connection & software watchdog timeout
- Malformed command rejection
- Unknown command rejection
- Safe STOP behavior on disconnections and E-stop
- Secret isolation (token never in logs or status)
- Hardware inactivity verification (hardware_active == False)
"""

from __future__ import annotations

import io
import logging
import os
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import socketio

from app.robot_agent.config import (
    ConfigurationError,
    RobotAgentConfig,
    load_agent_config,
)
from app.robot_agent.connection_state import ConnectionState
from app.robot_agent.safety_authority import LocalSafetyAuthority, NavCommand
from app.robot_agent.watchdog import SoftwareWatchdog
from app.robot_agent.client import RobotAgent, TokenRedactionFilter


class TestRobotAgentConfig(unittest.TestCase):
    """Verifies configuration loading, validation, and secret masking."""

    def setUp(self):
        self.old_env = os.environ.copy()

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.old_env)

    def test_missing_token_fails_closed(self):
        """Missing SPOTMICRO_OPERATOR_API_TOKEN must raise ConfigurationError."""
        os.environ["SPOTMICRO_BACKEND_URL"] = "https://spotmicro-backend.onrender.com"
        os.environ.pop("SPOTMICRO_OPERATOR_API_TOKEN", None)

        with self.assertRaises(ConfigurationError) as ctx:
            load_agent_config()
        self.assertIn("Missing or empty SPOTMICRO_OPERATOR_API_TOKEN", str(ctx.exception))

    def test_missing_backend_url_fails_closed(self):
        """Missing SPOTMICRO_BACKEND_URL must raise ConfigurationError."""
        os.environ.pop("SPOTMICRO_BACKEND_URL", None)
        os.environ["SPOTMICRO_OPERATOR_API_TOKEN"] = "sample-test-token"

        with self.assertRaises(ConfigurationError) as ctx:
            load_agent_config()
        self.assertIn("Missing or invalid SPOTMICRO_BACKEND_URL", str(ctx.exception))

    def test_invalid_backend_url_scheme_fails_closed(self):
        """Backend URL must have http:// or https:// scheme."""
        with self.assertRaises(ConfigurationError):
            RobotAgentConfig(backend_url="ftp://invalid.com", operator_token="test-token")

        with self.assertRaises(ConfigurationError):
            RobotAgentConfig(backend_url="invalid-no-scheme", operator_token="test-token")

    def test_load_dotenv_if_present(self):
        """Verify that _load_dotenv_if_present safely extracts config from a .env file."""
        import tempfile
        with tempfile.NamedTemporaryFile("w", delete=False) as tf:
            tf.write("SPOTMICRO_BACKEND_URL=https://spotmicro-backend.onrender.com\n")
            tf.write("SPOTMICRO_OPERATOR_API_TOKEN=sample-dotenv-token\n")
            tmp_path = tf.name

        try:
            with patch.dict(os.environ, {}, clear=True):
                cfg = load_agent_config(dotenv_path=tmp_path)
                self.assertEqual(cfg.backend_url, "https://spotmicro-backend.onrender.com")
                self.assertEqual(cfg.operator_token, "sample-dotenv-token")
        finally:
            os.unlink(tmp_path)

    def test_valid_configuration_and_secret_masking(self):
        """Valid configuration must redact token in __repr__ and provide correct auth structures."""
        secret_token = "ultra-secret-operator-token-987"
        cfg = RobotAgentConfig(
            backend_url="https://spotmicro-backend.onrender.com",
            operator_token=secret_token,
            watchdog_timeout_s=3.0,
            heartbeat_interval_s=1.0,
        )

        self.assertEqual(cfg.backend_url, "https://spotmicro-backend.onrender.com")
        self.assertEqual(cfg.operator_token, secret_token)
        self.assertEqual(cfg.watchdog_timeout_s, 3.0)
        self.assertEqual(cfg.heartbeat_interval_s, 1.0)

        # Confirm token is NEVER in string or repr
        repr_str = repr(cfg)
        str_val = str(cfg)
        self.assertNotIn(secret_token, repr_str)
        self.assertIn("***REDACTED***", repr_str)
        self.assertNotIn(secret_token, str_val)
        self.assertIn("***REDACTED***", str_val)

        # Confirm auth structures
        self.assertEqual(cfg.auth_headers, {"Authorization": f"Bearer {secret_token}"})
        self.assertEqual(cfg.socketio_auth, {"token": secret_token})


class TestSoftwareWatchdog(unittest.TestCase):
    """Verifies software-only watchdog behavior and callback triggering."""

    def test_watchdog_timeout_triggers_callback(self):
        callback_called = False

        def on_stale():
            nonlocal callback_called
            callback_called = True

        watchdog = SoftwareWatchdog(timeout_s=0.1, on_stale_callback=on_stale)
        self.assertFalse(watchdog.is_stale)
        self.assertTrue(watchdog.check())

        # Wait until timeout expires
        time.sleep(0.15)
        self.assertFalse(watchdog.check())
        self.assertTrue(watchdog.is_stale)
        self.assertTrue(callback_called)

        # Feeding watchdog resets staleness
        watchdog.feed()
        self.assertFalse(watchdog.is_stale)
        self.assertTrue(watchdog.check())

    def test_watchdog_background_monitor(self):
        tripped = threading.Event()

        watchdog = SoftwareWatchdog(
            timeout_s=0.1,
            on_stale_callback=lambda: tripped.set(),
            check_interval_s=0.05,
        )
        watchdog.start()
        try:
            self.assertTrue(tripped.wait(timeout=0.5))
            self.assertTrue(watchdog.is_stale)
        finally:
            watchdog.stop()


class TestLocalSafetyAuthority(unittest.TestCase):
    """Verifies high-level command filtering, emergency stop, and hardware decoupling."""

    def setUp(self):
        self.safety = LocalSafetyAuthority()

    def test_hardware_strictly_inactive(self):
        """hardware_active must be False in Stage 2.1."""
        self.assertFalse(self.safety.hardware_active)

    def test_valid_commands_accepted(self):
        """Only high-level commands are accepted and mapped."""
        valid_cases = [
            ("FORWARD", NavCommand.FORWARD),
            ("forward", NavCommand.FORWARD),
            ("BACKWARD", NavCommand.BACKWARD),
            ("backward", NavCommand.BACKWARD),
            ("TURN_LEFT", NavCommand.TURN_LEFT),
            ("left", NavCommand.TURN_LEFT),
            ("TURN_RIGHT", NavCommand.TURN_RIGHT),
            ("right", NavCommand.TURN_RIGHT),
            ("STOP", NavCommand.STOP),
            ("stop", NavCommand.STOP),
            ("stand", NavCommand.WAIT),
            ("sit", NavCommand.STOP),
            ("AVOID_OBSTACLE", NavCommand.AVOID_OBSTACLE),
            ("WAIT", NavCommand.WAIT),
        ]

        for raw_cmd, expected_enum in valid_cases:
            with self.subTest(raw_cmd=raw_cmd):
                ok, cmd, msg = self.safety.process_command({"command": raw_cmd})
                self.assertTrue(ok)
                self.assertEqual(cmd, expected_enum)
                self.assertEqual(self.safety.current_command, expected_enum)

    def test_malformed_command_enters_safe_stop(self):
        """Malformed payloads (non-dict, non-string, missing command) must force STOP."""
        malformed_inputs = [
            None,
            {},
            {"command": None},
            {"command": 123},
            {"command": ["forward"]},
            {"wrong_key": "forward"},
            "",
            12345,
        ]

        for bad_input in malformed_inputs:
            with self.subTest(bad_input=bad_input):
                ok, cmd, msg = self.safety.process_command(bad_input)
                self.assertFalse(ok)
                self.assertEqual(cmd, NavCommand.STOP)
                self.assertEqual(self.safety.current_command, NavCommand.STOP)
                self.assertTrue(self.safety.is_safe_idle)

    def test_unknown_command_enters_safe_stop(self):
        """Unknown or disallowed commands must force STOP."""
        unknown_inputs = [
            "FLY",
            "JUMP",
            "ROLLOVER",
            "ATTACK",
            {"command": "DISARM"},
            {"command": "SET_SERVO_ANGLE_90"},
        ]

        for bad_cmd in unknown_inputs:
            with self.subTest(bad_cmd=bad_cmd):
                ok, cmd, msg = self.safety.process_command(bad_cmd)
                self.assertFalse(ok)
                self.assertEqual(cmd, NavCommand.STOP)
                self.assertEqual(self.safety.current_command, NavCommand.STOP)
                self.assertTrue(self.safety.is_safe_idle)

    def test_emergency_stop_behavior(self):
        """Emergency Stop must lock in STOP and reject subsequent commands until reset."""
        # Start in forward motion
        self.safety.process_command("forward")
        self.assertEqual(self.safety.current_command, NavCommand.FORWARD)

        # Trigger E-stop
        self.safety.trigger_emergency_stop("Manual button pressed")
        self.assertTrue(self.safety.emergency_stop)
        self.assertEqual(self.safety.current_command, NavCommand.STOP)
        self.assertTrue(self.safety.is_safe_idle)

        # Attempting forward during E-stop must fail
        ok, cmd, msg = self.safety.process_command("forward")
        self.assertFalse(ok)
        self.assertEqual(cmd, NavCommand.STOP)
        self.assertIn("Emergency Stop is active", msg)

        # Resetting E-stop allows subsequent commands
        self.safety.reset_emergency_stop()
        self.assertFalse(self.safety.emergency_stop)
        ok, cmd, msg = self.safety.process_command("forward")
        self.assertTrue(ok)
        self.assertEqual(cmd, NavCommand.FORWARD)


class TestRobotAgentClient(unittest.TestCase):
    """Verifies end-to-end agent client, connection lifecycle, and secret protection."""

    def setUp(self):
        self.test_token = "secure-agent-test-token-456"
        self.config = RobotAgentConfig(
            backend_url="https://spotmicro-backend.onrender.com",
            operator_token=self.test_token,
            watchdog_timeout_s=0.2,
            heartbeat_interval_s=0.1,
        )

    def test_initial_state_disconnected(self):
        """Newly created agent must start in DISCONNECTED state with STOP command."""
        agent = RobotAgent(config=self.config)
        self.assertEqual(agent.state, ConnectionState.DISCONNECTED)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertFalse(agent.is_connected)

    def test_missing_config_initializes_to_error_and_safe_idle(self):
        """Initializing RobotAgent with missing environment variables fails closed."""
        with patch.dict(os.environ, {}, clear=True):
            agent = RobotAgent(config=None)
            self.assertEqual(agent.state, ConnectionState.ERROR)
            self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
            self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_invalid_authentication_enters_error_and_safe_idle(self):
        """Simulated authentication rejection causes transition to ERROR and STOP."""
        mock_sio = MagicMock(spec=socketio.Client)
        mock_sio.connected = False
        mock_sio.connect.side_effect = socketio.exceptions.ConnectionError("Authentication required")

        agent = RobotAgent(config=self.config, sio_client=mock_sio)
        success = agent.connect()

        self.assertFalse(success)
        self.assertEqual(agent.state, ConnectionState.ERROR)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_backend_unavailable_enters_error_and_safe_idle(self):
        """Network unreachable or connection timeout transitions to ERROR and STOP."""
        mock_sio = MagicMock(spec=socketio.Client)
        mock_sio.connected = False
        mock_sio.connect.side_effect = Exception("Connection refused / network unreachable")

        agent = RobotAgent(config=self.config, sio_client=mock_sio)
        success = agent.connect()

        self.assertFalse(success)
        self.assertEqual(agent.state, ConnectionState.ERROR)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_stale_connection_watchdog_transition(self):
        """When the watchdog expires, agent transitions to SAFE_IDLE and STOP."""
        agent = RobotAgent(config=self.config)
        agent._set_state(ConnectionState.AUTHENTICATED)
        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        # Trigger watchdog timeout directly
        agent._on_watchdog_stale()

        self.assertEqual(agent.state, ConnectionState.SAFE_IDLE)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_disconnect_forces_safe_stop(self):
        """Calling disconnect() transitions to DISCONNECTED and STOP."""
        mock_sio = MagicMock(spec=socketio.Client)
        mock_sio.connected = True

        agent = RobotAgent(config=self.config, sio_client=mock_sio)
        agent._set_state(ConnectionState.AUTHENTICATED)
        agent.safety_authority.process_command("forward")

        agent.disconnect()

        self.assertEqual(agent.state, ConnectionState.DISCONNECTED)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        mock_sio.disconnect.assert_called_once()

    def test_token_never_appears_in_logs(self):
        """Verifies that the secret operator token is never logged by the agent."""
        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.DEBUG)

        agent_logger = logging.getLogger("SpotMicroRobotAgent")
        agent_logger.addHandler(handler)
        agent_logger.setLevel(logging.DEBUG)

        try:
            agent = RobotAgent(config=self.config)
            # Log various messages including config representation
            agent_logger.info(f"Loaded config: {agent.config}")
            agent_logger.warning("Authentication failure on backend")
            agent.safety_authority.process_command("forward")
            agent.safety_authority.trigger_emergency_stop("Test E-stop")
            agent.disconnect()

            captured_logs = log_stream.getvalue()
            # Assert that the real test token NEVER appears in the log stream
            self.assertNotIn(self.test_token, captured_logs)
            self.assertIn("***REDACTED***", captured_logs)
        finally:
            agent_logger.removeHandler(handler)

    def test_status_output_contains_no_secrets(self):
        """Agent diagnostic status must not contain operator token."""
        agent = RobotAgent(config=self.config)
        status = agent.get_status()

        self.assertIn("connection_state", status)
        self.assertIn("safety_authority", status)
        self.assertNotIn("token", status)
        self.assertNotIn(self.test_token, str(status))

    def test_hardware_strictly_inactive_and_not_imported(self):
        """Confirms that hardware_active remains False and hardware modules are not imported."""
        agent = RobotAgent(config=self.config)
        self.assertFalse(agent.safety_authority.hardware_active)

        # Confirm no hardware modules are in sys.modules
        for forbidden in ("adafruit_pca9685", "board", "busio", "RPi.GPIO"):
            self.assertNotIn(forbidden, sys.modules)


class TestFailureAndRecoverySafety(unittest.TestCase):
    """
    Stage 2.4 Failure & Recovery Safety Verification Suite
    Verifies that the Robot Agent fails safely (STOP + SAFE_IDLE) and
    recovers safely without EVER issuing automatic movement commands.
    """

    def setUp(self):
        self.test_token = "valid-stage24-test-token"
        self.config = RobotAgentConfig(
            backend_url="https://spotmicro-backend.onrender.com",
            operator_token=self.test_token,
            watchdog_timeout_s=0.2,
            heartbeat_interval_s=0.1,
        )

    def _create_agent_with_mock_sio(self, connected: bool = True):
        mock_sio = MagicMock(spec=socketio.Client)
        mock_sio.connected = connected
        handlers = {}

        def mock_on(event):
            def decorator(func):
                handlers[event] = func
                return func
            return decorator

        mock_sio.on.side_effect = mock_on
        agent = RobotAgent(config=self.config, sio_client=mock_sio)
        return agent, mock_sio, handlers

    def test_scenario_a_normal_connection_and_telemetry(self):
        """
        Scenario A: Normal connection & telemetry processing.
        Verify DISCONNECTED -> CONNECTING -> CONNECTED -> AUTHENTICATED.
        Verify heartbeat/telemetry feeds watchdog and robot remains in SAFE_IDLE / STOP.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        self.assertEqual(agent.state, ConnectionState.DISCONNECTED)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # Connect
        connected = agent.connect(wait_timeout=2.0)
        self.assertTrue(connected)
        self.assertEqual(agent.state, ConnectionState.AUTHENTICATED)
        self.assertTrue(agent.is_connected)

        # Receive simulation telemetry
        self.assertIn("telemetry_update", handlers)
        prev_kick = agent.watchdog.last_kick_time
        time.sleep(0.02)
        handlers["telemetry_update"]({"battery_voltage": 12.4, "mode": "simulation"})
        self.assertGreater(agent.watchdog.last_kick_time, prev_kick)
        self.assertFalse(agent.watchdog.is_stale)

        # Robot must remain in safe state (STOP)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_scenario_b_socketio_disconnect_triggers_stop_and_safe_idle(self):
        """
        Scenario B: Unexpected Socket.IO disconnect.
        Agent must enter DISCONNECTED, trigger STOP, enter SAFE_IDLE,
        and generate no movement.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect(wait_timeout=2.0)

        # Robot was commanded to forward during active session
        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        # Invoke disconnect event callback
        self.assertIn("disconnect", handlers)
        handlers["disconnect"]()

        # State must be DISCONNECTED, SafetyAuthority must be SAFE_IDLE / STOP
        self.assertEqual(agent.state, ConnectionState.DISCONNECTED)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_scenario_c_watchdog_timeout_and_recovery(self):
        """
        Scenario C: Stale heartbeat/telemetry triggers watchdog timeout.
        Watchdog expires -> STOP + SAFE_IDLE.
        Restoring fresh heartbeat -> watchdog recovers, but robot does NOT automatically move.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect(wait_timeout=2.0)

        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        # Simulate watchdog timeout (stale telemetry)
        agent._on_watchdog_stale()

        self.assertEqual(agent.state, ConnectionState.SAFE_IDLE)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

        # Restore fresh heartbeat
        agent.watchdog.feed()
        self.assertFalse(agent.watchdog.is_stale)

        # Recovery must NEVER automatically resume FORWARD motion!
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_scenario_d_backend_interruption_and_recovery(self):
        """
        Scenario D: Backend becomes unavailable and subsequently recovers.
        Failure -> STOP + SAFE_IDLE.
        Recovery -> Reconnect succeeds, but robot remains safely in STOP.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect(wait_timeout=2.0)
        agent.safety_authority.process_command("forward")

        # Simulate backend interruption error event
        self.assertIn("connect_error", handlers)
        handlers["connect_error"]("Backend server 503 / unreachable")

        self.assertEqual(agent.state, ConnectionState.ERROR)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

        # Reconnection after backend recovery
        self.assertIn("connect", handlers)
        handlers["connect"]()

        self.assertEqual(agent.state, ConnectionState.AUTHENTICATED)
        # Robot MUST remain STOP (no automatic movement upon reconnect)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_scenario_e_network_failure_and_recovery(self):
        """
        Scenario E: Controlled mocked network connection failure.
        Failure -> STOP + SAFE_IDLE.
        Recovery -> reconnects safely, zero automatic motion.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=False)
        # Mock network failure on connect
        mock_sio.connect.side_effect = socketio.exceptions.ConnectionError("Network interface unreachable")

        success = agent.connect(wait_timeout=1.0)

        self.assertFalse(success)
        self.assertEqual(agent.state, ConnectionState.ERROR)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

        # Network recovers
        mock_sio.connect.side_effect = None
        mock_sio.connected = True
        success_reconnect = agent.connect(wait_timeout=1.0)

        self.assertTrue(success_reconnect)
        self.assertEqual(agent.state, ConnectionState.AUTHENTICATED)
        # Robot remains safely stopped
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_scenario_f_invalid_token_fails_safely(self):
        """
        Scenario F: Invalid/expired operator authentication.
        Auth failure -> ERROR, STOP, SAFE_IDLE, zero movement, token redacted.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=False)
        mock_sio.connect.side_effect = socketio.exceptions.ConnectionError("401 Unauthorized: Invalid token")

        success = agent.connect()

        self.assertFalse(success)
        self.assertEqual(agent.state, ConnectionState.ERROR)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.is_connected)

    def test_scenario_g_process_restart_starts_in_safe_idle(self):
        """
        Scenario G: Robot Agent process restart in software.
        Must initialize in DISCONNECTED, current command STOP, is_safe_idle True.
        Must NEVER start with a movement command.
        """
        # Simulate fresh process startup
        fresh_agent = RobotAgent(config=self.config)
        self.assertEqual(fresh_agent.state, ConnectionState.DISCONNECTED)
        self.assertEqual(fresh_agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(fresh_agent.safety_authority.is_safe_idle)
        self.assertFalse(fresh_agent.safety_authority.hardware_active)
        self.assertFalse(fresh_agent.is_connected)

    def test_scenario_h_recovery_never_issues_automatic_movement(self):
        """
        Scenario H: Comprehensive verification across all recovery flows.
        Verifies that after any failure/recovery cycle, the safety state
        remains STOP until an explicit valid command is received.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Step 1: Forward movement commanded
        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        # Step 2: Failure occurs (Watchdog timeout)
        agent._on_watchdog_stale()
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

        # Step 3: Watchdog feeds resume (Recovery 1)
        agent.watchdog.feed()
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # Step 4: Disconnect occurs
        agent.disconnect()
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # Step 5: Reconnect occurs (Recovery 2)
        agent.connect()
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

        # Movement ONLY resumes when explicit new command arrives
        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

    def test_step_3_estop_persistence_across_reconnect_watchdog_and_restart(self):
        """
        Step 3: E-stop persistence verification.
        Triggering E-stop locks the robot in STOP.
        Reconnecting, watchdog recovery, or process restarts MUST NOT clear E-stop.
        Only explicit reset_emergency_stop() clears it.
        """
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Trigger Emergency Stop
        agent.safety_authority.trigger_emergency_stop("Manual trigger")
        self.assertTrue(agent.safety_authority.emergency_stop)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # 1. Reconnection must NOT clear E-stop
        self.assertIn("connect", handlers)
        handlers["connect"]()
        self.assertTrue(agent.safety_authority.emergency_stop)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # 2. Watchdog feed/recovery must NOT clear E-stop
        agent.watchdog.feed()
        self.assertTrue(agent.safety_authority.emergency_stop)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # 3. New commands during E-stop must be BLOCKED
        ok, cmd, msg = agent.safety_authority.process_command("forward")
        self.assertFalse(ok)
        self.assertEqual(cmd, NavCommand.STOP)
        self.assertIn("Emergency Stop is active", msg)

        # 4. Only explicit reset clears it
        agent.safety_authority.reset_emergency_stop()
        self.assertFalse(agent.safety_authority.emergency_stop)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_step_4_hardware_isolation_throughout_all_failure_scenarios(self):
        """
        Step 4: Hardware isolation throughout all failure & recovery testing.
        Confirms hardware_active == False and zero hardware library imports.
        """
        agent = RobotAgent(config=self.config)
        self.assertFalse(agent.safety_authority.hardware_active)

        # Trigger multiple state transitions
        agent.safety_authority.process_command("forward")
        agent.safety_authority.enter_safe_stop("Test")
        agent.safety_authority.trigger_emergency_stop("Test")
        agent.safety_authority.reset_emergency_stop()

        self.assertFalse(agent.safety_authority.hardware_active)
        for forbidden in ("adafruit_pca9685", "board", "busio", "RPi.GPIO"):
            self.assertNotIn(forbidden, sys.modules)


class TestStage25CommandBridge(unittest.TestCase):
    """
    SpotMicro Stage 2.5 — Backend -> Raspberry Pi Command Bridge Test Suite
    Verifies software-only command bridge between hardened Render backend and
    Raspberry Pi 5 Robot Agent across all 14 safety and failure cases.
    """

    def setUp(self):
        self.old_env = os.environ.copy()
        self.test_token = os.environ.get("OPERATOR_API_TOKEN") or "stage25-test-operator-token-777"
        os.environ["OPERATOR_API_TOKEN"] = self.test_token

        from web.backend.server import (
            app,
            clear_connected_robot_agents,
            get_connected_robot_agent_count,
            is_robot_agent_connected,
            register_robot_agent_session,
            simulator,
            unregister_robot_agent_session,
        )
        from web.backend.security import rate_limiter

        self.server_app = app
        self.simulator = simulator
        self.rate_limiter = rate_limiter
        self.register_robot_agent = register_robot_agent_session
        self.unregister_robot_agent = unregister_robot_agent_session
        self.clear_robot_agents = clear_connected_robot_agents
        self.is_robot_connected = is_robot_agent_connected
        self.get_robot_count = get_connected_robot_agent_count

        self.rate_limiter.reset()
        self.clear_robot_agents()
        self.simulator.reset_emergency_stop()
        self.simulator.apply_scenario("nominal")

        self.client = self.server_app.test_client()
        self.auth_headers = {"Authorization": f"Bearer {self.test_token}"}

        self.config = RobotAgentConfig(
            backend_url="https://spotmicro-backend.onrender.com",
            operator_token=self.test_token,
            watchdog_timeout_s=0.2,
            heartbeat_interval_s=0.1,
        )

    def tearDown(self):
        self.clear_robot_agents()
        self.simulator.reset_emergency_stop()
        os.environ.clear()
        os.environ.update(self.old_env)

    def _create_agent_with_mock_sio(self, connected: bool = True):
        mock_sio = MagicMock(spec=socketio.Client)
        mock_sio.connected = connected
        handlers = {}

        def mock_on(event):
            def decorator(func):
                handlers[event] = func
                return func
            return decorator

        mock_sio.on.side_effect = mock_on
        agent = RobotAgent(config=self.config, sio_client=mock_sio)
        return agent, mock_sio, handlers

    def test_case_01_pi_connected_forward_forwarded(self):
        """Case 1: Pi connected -> FORWARD command is forwarded and accepted by Pi."""
        self.register_robot_agent("session-pi5-case1")
        self.assertTrue(self.is_robot_connected())

        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        res = self.client.post("/api/commands", json={"command": "forward"}, headers=self.auth_headers)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("command"), "forward")
        self.assertTrue(data.get("robot_connected"))
        self.assertTrue(data.get("robot_executed"))
        self.assertEqual(data.get("robot_delivery"), "forwarded")

        # Pi receives forwarded command payload through on_command
        self.assertIn("command_executed", handlers)
        handlers["command_executed"](data)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)
        self.assertFalse(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_case_02_pi_connected_stop_forwarded(self):
        """Case 2: Pi connected -> STOP command is forwarded and places Pi in SAFE_IDLE / STOP."""
        self.register_robot_agent("session-pi5-case2")
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Robot was in forward motion
        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        res = self.client.post("/api/commands", json={"command": "stop"}, headers=self.auth_headers)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("command"), "stop")
        self.assertTrue(data.get("robot_connected"))
        self.assertTrue(data.get("robot_executed"))
        self.assertEqual(data.get("robot_delivery"), "forwarded")

        # Forwarded to agent
        handlers["command_executed"](data)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_case_03_pi_disconnected_movement_not_executed(self):
        """Case 3: Pi disconnected -> movement command is NOT executed on Pi (remains in STOP)."""
        self.clear_robot_agents()
        self.assertFalse(self.is_robot_connected())

        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=False)
        self.assertEqual(agent.state, ConnectionState.DISCONNECTED)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # Dashboard issues FORWARD command
        res = self.client.post("/api/commands", json={"command": "forward"}, headers=self.auth_headers)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Pi is disconnected so command cannot be delivered to Pi
        # Pi remains safely in STOP / SAFE_IDLE
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_case_04_pi_disconnected_backend_does_not_report_successful_execution(self):
        """Case 4: Pi disconnected -> backend does NOT report successful robot execution."""
        self.clear_robot_agents()
        self.assertFalse(self.is_robot_connected())

        res = self.client.post("/api/commands", json={"command": "forward"}, headers=self.auth_headers)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        # Backend must clearly state robot execution did NOT occur
        self.assertFalse(data.get("robot_connected"))
        self.assertFalse(data.get("robot_executed"))
        self.assertEqual(data.get("robot_delivery"), "unavailable")

    def test_case_05_auth_failure_command_rejected(self):
        """Case 5: Authentication failure -> command rejected (401), not forwarded."""
        self.register_robot_agent("session-pi5-case5")
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Missing token
        res_no_auth = self.client.post("/api/commands", json={"command": "forward"})
        self.assertEqual(res_no_auth.status_code, 401)
        self.assertEqual(res_no_auth.get_json().get("error"), "Authentication required")

        # Invalid token
        res_bad_auth = self.client.post(
            "/api/commands",
            json={"command": "forward"},
            headers={"Authorization": "Bearer invalid-tampered-token"},
        )
        self.assertEqual(res_bad_auth.status_code, 401)
        self.assertEqual(res_bad_auth.get_json().get("error"), "Authentication required")

        # Robot agent remained untouched in STOP
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_case_06_invalid_command_rejected(self):
        """Case 6: Invalid command -> rejected by backend and rejected by Pi Safety Authority."""
        self.register_robot_agent("session-pi5-case6")
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Unknown command string to backend
        res_invalid = self.client.post("/api/commands", json={"command": "jump"}, headers=self.auth_headers)
        self.assertEqual(res_invalid.status_code, 400)
        self.assertEqual(res_invalid.get_json().get("status"), "rejected")

        # Non-string command to backend
        res_nonstring = self.client.post("/api/commands", json={"command": 12345}, headers=self.auth_headers)
        self.assertEqual(res_nonstring.status_code, 400)

        # Direct test on LocalSafetyAuthority
        ok, cmd, msg = agent.safety_authority.process_command({"command": "invalid_command_xyz"})
        self.assertFalse(ok)
        self.assertEqual(cmd, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_case_07_emergency_stop_pi_remains_stop_safe_idle(self):
        """Case 7: Emergency stop -> Pi remains STOP/SAFE_IDLE; commands blocked."""
        self.register_robot_agent("session-pi5-case7")
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Engage Emergency Stop on backend
        res_estop = self.client.post("/api/emergency-stop", headers=self.auth_headers)
        self.assertEqual(res_estop.status_code, 200)

        # Pi receives emergency_stop_triggered event
        self.assertIn("emergency_stop_triggered", handlers)
        handlers["emergency_stop_triggered"](res_estop.get_json())
        self.assertTrue(agent.safety_authority.emergency_stop)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

        # Movement command on backend must be rejected
        res_move = self.client.post("/api/commands", json={"command": "forward"}, headers=self.auth_headers)
        self.assertEqual(res_move.status_code, 400)
        self.assertIn("emergency stop is active", res_move.get_json().get("reason", "").lower())

        # LocalSafetyAuthority rejects any direct movement attempt under E-stop
        ok, cmd, msg = agent.safety_authority.process_command("forward")
        self.assertFalse(ok)
        self.assertEqual(cmd, NavCommand.STOP)
        self.assertIn("Emergency Stop is active", msg)

        # STOP command is always accepted under E-Stop and keeps robot in STOP
        ok_stop, cmd_stop, msg_stop = agent.safety_authority.process_command("stop")
        self.assertTrue(ok_stop)
        self.assertEqual(cmd_stop, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_case_08_pi_disconnect_after_receiving_command_safe_stop(self):
        """Case 8: Pi disconnect after receiving a command -> immediately halts to safe STOP."""
        self.register_robot_agent("session-pi5-case8")
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Robot moving forward
        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        # Disconnect event occurs
        self.assertIn("disconnect", handlers)
        handlers["disconnect"]()

        # Robot must immediately drop to STOP and SAFE_IDLE
        self.assertEqual(agent.state, ConnectionState.DISCONNECTED)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_case_09_pi_reconnect_remains_stop(self):
        """Case 9: Pi reconnect -> remains in STOP (never resumes previous movement)."""
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        # Prior movement before disconnect
        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        # Disconnection halts robot
        handlers["disconnect"]()
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

        # Reconnection event
        handlers["connect"]()
        self.assertEqual(agent.state, ConnectionState.AUTHENTICATED)

        # Robot MUST remain STOP
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_case_10_watchdog_timeout_safe_idle_stop(self):
        """Case 10: Watchdog timeout -> transitions to SAFE_IDLE / STOP."""
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)
        agent.connect()

        agent.safety_authority.process_command("forward")
        self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

        # Watchdog timeout fires
        agent._on_watchdog_stale()

        self.assertEqual(agent.state, ConnectionState.SAFE_IDLE)
        self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
        self.assertTrue(agent.safety_authority.is_safe_idle)
        self.assertFalse(agent.safety_authority.hardware_active)

    def test_case_11_repeated_disconnect_reconnect_no_automatic_movement(self):
        """Case 11: Repeated disconnect/reconnect cycles -> zero automatic movement."""
        agent, mock_sio, handlers = self._create_agent_with_mock_sio(connected=True)

        for cycle in range(3):
            # Connect
            handlers["connect"]()
            self.assertEqual(agent.state, ConnectionState.AUTHENTICATED)
            self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)

            # Move
            agent.safety_authority.process_command("forward")
            self.assertEqual(agent.safety_authority.current_command, NavCommand.FORWARD)

            # Disconnect
            handlers["disconnect"]()
            self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
            self.assertTrue(agent.safety_authority.is_safe_idle)

            # Reconnect -> must be STOP
            handlers["connect"]()
            self.assertEqual(agent.safety_authority.current_command, NavCommand.STOP)
            self.assertTrue(agent.safety_authority.is_safe_idle)

    def test_case_12_hardware_isolation_no_hardware_modules_imported(self):
        """Case 12: Hardware isolation -> no physical hardware modules imported."""
        for forbidden in ("adafruit_pca9685", "board", "busio", "RPi.GPIO"):
            self.assertNotIn(forbidden, sys.modules)

    def test_case_13_backend_remains_simulation_mode(self):
        """Case 13: Backend remains in simulation mode throughout operations."""
        res_health = self.client.get("/api/health")
        self.assertEqual(res_health.status_code, 200)
        hdata = res_health.get_json()
        self.assertEqual(hdata.get("mode"), "simulation")
        self.assertEqual(hdata.get("status"), "ok")
        self.assertFalse(hdata.get("hardware_active"))

        res_status = self.client.get("/api/status")
        self.assertEqual(res_status.status_code, 200)
        sdata = res_status.get_json()
        self.assertEqual(sdata.get("mode"), "simulation")
        self.assertTrue(sdata.get("simulated"))

    def test_case_14_hardware_active_remains_false(self):
        """Case 14: hardware_active strictly remains False across all components."""
        agent = RobotAgent(config=self.config)
        self.assertFalse(agent.safety_authority.hardware_active)
        self.assertFalse(self.simulator.state.hardware_active)

        res_config = self.client.get("/api/config")
        self.assertEqual(res_config.status_code, 200)
        cdata = res_config.get_json()
        self.assertFalse(cdata.get("hardware_active"))


if __name__ == "__main__":
    unittest.main()
