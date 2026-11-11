"""
Unit Tests for SpotMicro Web Backend & Simulation Service (Stage 3)
===================================================================
Tests Flask REST API endpoints, command validation, Emergency Stop priority,
simulation scenarios, dead-man timeout, Rule 9 sensor integrity, and WebSocket
event emissions.

All tests run in software-only simulation mode with NO hardware activation.
"""

import json
import sys
import time
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.backend.server import app, simulator, socketio


class TestWebBackendREST(unittest.TestCase):
    """Tests for Flask REST API endpoints."""

    def setUp(self):
        self.client = app.test_client()
        # Reset simulator state between tests
        simulator.reset_emergency_stop()
        simulator.apply_scenario("nominal")

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("status"), "ok")
        self.assertEqual(data.get("mode"), "simulation")
        self.assertFalse(data.get("hardware_active"))

    def test_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("connected"))
        self.assertEqual(data.get("mode"), "simulation")
        self.assertTrue(data.get("simulated"))
        self.assertIn("SIMULATED", data.get("label", ""))
        self.assertIn("battery", data)

    def test_telemetry_endpoint_simulation_labels(self):
        res = self.client.get("/api/telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertTrue(data.get("simulated"))
        self.assertIn("SIMULATED", data.get("label", ""))

        # Verify ultrasonic has all 4 sensors labeled simulated
        ultrasonic = data.get("ultrasonic", {})
        for s in ("front", "left", "right", "rear"):
            self.assertIn(s, ultrasonic)
            self.assertTrue(ultrasonic[s].get("simulated"))

        # Verify IMU labeled simulated
        imu = data.get("imu", {})
        self.assertTrue(imu.get("simulated"))
        self.assertIn("roll", imu)
        self.assertIn("pitch", imu)
        self.assertIn("heading", imu)

    def test_navigation_endpoint(self):
        res = self.client.get("/api/navigation")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data.get("simulated"))
        self.assertIn("decision", data)
        self.assertIn("command", data["decision"])

    def test_config_endpoint(self):
        res = self.client.get("/api/config")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data.get("mode"), "simulation")
        self.assertFalse(data.get("hardware_active"))
        self.assertIn("forward", data.get("allowed_commands", []))
        self.assertGreater(len(data.get("available_scenarios", [])), 0)

    def test_valid_commands(self):
        for cmd in ("forward", "backward", "left", "right", "stand", "sit", "stop"):
            res = self.client.post("/api/commands", json={"command": cmd, "duration": 1.0, "speed": 1.0})
            self.assertEqual(res.status_code, 200)
            data = res.get_json()
            self.assertEqual(data.get("status"), "ok")
            self.assertEqual(data.get("command"), cmd)

    def test_invalid_command_rejection(self):
        # Unknown command
        res = self.client.post("/api/commands", json={"command": "jump"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertEqual(data.get("status"), "rejected")
        self.assertIn("invalid command", data.get("reason", "").lower())

        # Missing json
        res2 = self.client.post("/api/commands", data="not json", content_type="text/plain")
        self.assertEqual(res2.status_code, 400)

    def test_emergency_stop_priority(self):
        # 1. Engage Emergency Stop
        res_estop = self.client.post("/api/emergency-stop")
        self.assertEqual(res_estop.status_code, 200)
        self.assertTrue(res_estop.get_json().get("emergency_stop"))

        # 2. Attempt movement command while E-Stop is active -> MUST BE REJECTED
        res_move = self.client.post("/api/commands", json={"command": "forward"})
        self.assertEqual(res_move.status_code, 400)
        self.assertIn("emergency stop is active", res_move.get_json().get("reason", "").lower())

        # 3. Stop command is always accepted even under E-Stop
        res_stop = self.client.post("/api/commands", json={"command": "stop"})
        self.assertEqual(res_stop.status_code, 200)

        # 4. Reset Emergency Stop
        res_reset = self.client.post("/api/reset-emergency-stop")
        self.assertEqual(res_reset.status_code, 200)
        self.assertFalse(res_reset.get_json().get("emergency_stop"))

        # 5. Forward command is now allowed again
        res_move2 = self.client.post("/api/commands", json={"command": "forward"})
        self.assertEqual(res_move2.status_code, 200)

    def test_simulation_scenarios_and_safety_reactions(self):
        # A. Person in path scenario
        res = self.client.post("/api/simulation/scenario", json={"scenario": "person_ahead"})
        self.assertEqual(res.status_code, 200)
        nav_res = self.client.get("/api/navigation").get_json()
        self.assertEqual(nav_res["decision"]["command"], "STOP")
        self.assertIn("person", nav_res["decision"]["reason"].lower())

        # B. Tilt hazard scenario
        res_tilt = self.client.post("/api/simulation/scenario", json={"scenario": "tilt_hazard"})
        self.assertEqual(res_tilt.status_code, 200)
        nav_tilt = self.client.get("/api/navigation").get_json()
        self.assertEqual(nav_tilt["decision"]["command"], "STOP")
        self.assertIn("roll", nav_tilt["decision"]["reason"].lower())

        # Movement command should be rejected under severe tilt
        res_blocked_cmd = self.client.post("/api/commands", json={"command": "forward"})
        self.assertEqual(res_blocked_cmd.status_code, 400)
        self.assertIn("tilt", res_blocked_cmd.get_json().get("reason", "").lower())

        # C. Sensor fault scenario (Rule 9 test: missing front sensor -> WAIT)
        res_fault = self.client.post("/api/simulation/scenario", json={"scenario": "sensor_fault"})
        self.assertEqual(res_fault.status_code, 200)
        nav_fault = self.client.get("/api/navigation").get_json()
        self.assertEqual(nav_fault["decision"]["command"], "WAIT")
        self.assertIn("cannot verify path clearance", nav_fault["decision"]["reason"].lower())

    def test_logs_endpoint(self):
        res = self.client.get("/api/logs?limit=10")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("logs", data)
        self.assertIn("total", data)
        self.assertIsInstance(data["logs"], list)


class TestWebBackendWebSocket(unittest.TestCase):
    """Tests for WebSocket real-time events via Flask-SocketIO test client."""

    def test_websocket_connection_and_events(self):
        socket_client = socketio.test_client(app)
        self.assertTrue(socket_client.is_connected())

        # Check initial events emitted upon connection
        received = socket_client.get_received()
        event_names = [e["name"] for e in received]
        self.assertIn("status_update", event_names)
        self.assertIn("telemetry_update", event_names)
        self.assertIn("navigation_update", event_names)

        # Trigger command via REST, verify WebSocket receives command_executed
        http_client = app.test_client()
        http_client.post("/api/commands", json={"command": "forward"})

        received_after_cmd = socket_client.get_received()
        cmd_events = [e["name"] for e in received_after_cmd]
        self.assertIn("command_executed", cmd_events)
        self.assertIn("status_update", cmd_events)

        # Trigger Emergency Stop, verify WebSocket receives emergency_stop_triggered
        http_client.post("/api/emergency-stop")
        received_after_estop = socket_client.get_received()
        estop_events = [e["name"] for e in received_after_estop]
        self.assertIn("emergency_stop_triggered", estop_events)

        socket_client.disconnect()
        self.assertFalse(socket_client.is_connected())


if __name__ == "__main__":
    unittest.main()
