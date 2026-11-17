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


if __name__ == "__main__":
    unittest.main()
