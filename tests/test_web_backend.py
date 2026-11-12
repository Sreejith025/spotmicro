"""
Unit Tests for SpotMicro Web Backend & Simulation Service (Stage 3)
===================================================================
Tests Flask REST API endpoints, command validation, Emergency Stop priority,
simulation scenarios, dead-man timeout, Rule 9 sensor integrity, and WebSocket
event emissions.

All tests run in software-only simulation mode with NO hardware activation.
"""

import json
import math
import sys
import time
import unittest
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from web.backend.server import app, simulator, socketio
from web.backend.simulation import Position


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
        self.assertIn("auto_mode_update", event_names)

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
        self.assertIn("auto_mode_update", estop_events)

        socket_client.disconnect()
        self.assertFalse(socket_client.is_connected())


class TestAutoModePipeline(unittest.TestCase):
    """
    Comprehensive tests for Autonomous Navigation Mode (Auto Mode).
    Verifies state transitions, perception pipeline, deterministic scenario reactions,
    emergency stop interlocks, and strict simulation isolation.
    """

    def setUp(self):
        self.client = app.test_client()
        # Reset to known clean state before each test
        simulator.reset_emergency_stop()
        simulator.disable_auto_mode()
        simulator.apply_scenario("nominal")

    def test_01_auto_mode_starts_disabled(self):
        """1. Auto Mode starts DISABLED."""
        res = self.client.get("/api/auto-mode")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data["enabled"])
        self.assertEqual(data["state"], "DISABLED")
        self.assertFalse(data["ai_detection"])
        self.assertFalse(data["auto_navigation"])
        self.assertFalse(data["movement_authority"])

    def test_02_and_03_enable_auto_mode_enters_active(self):
        """2. Enable Auto Mode & 3. Auto Mode enters ACTIVE."""
        res = self.client.post("/api/auto-mode/enable")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["enabled"])
        self.assertEqual(data["state"], "ACTIVE")
        self.assertTrue(data["ai_detection"])
        self.assertTrue(data["auto_navigation"])
        self.assertTrue(data["movement_authority"])

    def test_04_nominal_scenario_produces_forward(self):
        """4. Nominal scenario produces FORWARD."""
        self.client.post("/api/auto-mode/enable")
        res = self.client.post("/api/auto-mode/scenario", json={"scenario": "nominal"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "ACTIVE")
        self.assertEqual(data["current_command"], "FORWARD")
        self.assertEqual(data["safety_state"], "SAFE")
        self.assertTrue(data["movement_authority"])
        self.assertEqual(simulator.state.last_command, "forward")
        self.assertEqual(simulator.state.current_posture, "walking")

    def test_05_front_obstacle_produces_stop(self):
        """5. Front obstacle produces STOP."""
        self.client.post("/api/auto-mode/enable")
        res = self.client.post("/api/auto-mode/scenario", json={"scenario": "obstacle_front"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["current_command"], "STOP")
        self.assertEqual(data["state"], "STOPPED")
        self.assertEqual(data["safety_state"], "STOPPED")
        self.assertFalse(data["movement_authority"])
        self.assertEqual(simulator.state.last_command, "stop")

    def test_06_left_obstacle_produces_turn_right(self):
        """6. Left obstacle produces TURN_RIGHT."""
        self.client.post("/api/auto-mode/enable")
        res = self.client.post("/api/auto-mode/scenario", json={"scenario": "obstacle_left"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "ACTIVE")
        self.assertIn("RIGHT", data["current_command"])
        self.assertTrue(data["movement_authority"])
        self.assertEqual(simulator.state.last_command, "right")

    def test_07_right_obstacle_produces_turn_left(self):
        """7. Right obstacle produces TURN_LEFT."""
        self.client.post("/api/auto-mode/enable")
        res = self.client.post("/api/auto-mode/scenario", json={"scenario": "obstacle_right"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "ACTIVE")
        self.assertIn("LEFT", data["current_command"])
        self.assertTrue(data["movement_authority"])
        self.assertEqual(simulator.state.last_command, "left")

    def test_08_person_center_produces_stop(self):
        """8. Person center produces STOP."""
        self.client.post("/api/auto-mode/enable")
        res = self.client.post("/api/auto-mode/scenario", json={"scenario": "person_center"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["current_command"], "STOP")
        self.assertEqual(data["state"], "STOPPED")
        self.assertEqual(data["safety_state"], "STOPPED")
        self.assertFalse(data["movement_authority"])

    def test_09_stairs_ahead_produces_stop(self):
        """9. Stairs ahead produces STOP."""
        self.client.post("/api/auto-mode/enable")
        res = self.client.post("/api/auto-mode/scenario", json={"scenario": "stairs_ahead"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["current_command"], "STOP")
        self.assertEqual(data["state"], "STOPPED")
        self.assertEqual(data["safety_state"], "STOPPED")
        self.assertFalse(data["movement_authority"])

    def test_10_sensor_failure_produces_wait_stop(self):
        """10. Sensor failure produces WAIT/STOP (Rule 9)."""
        self.client.post("/api/auto-mode/enable")
        res = self.client.post("/api/auto-mode/scenario", json={"scenario": "sensor_failure"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn(data["current_command"], ("WAIT", "STOP"))
        self.assertEqual(data["state"], "STOPPED")
        self.assertFalse(data["movement_authority"])

    def test_11_and_12_emergency_stop_blocks_auto_mode(self):
        """11. Emergency Stop blocks Auto Mode & 12. Automatic command is rejected during E-Stop."""
        # Start in active auto mode
        self.client.post("/api/auto-mode/enable")
        self.assertEqual(simulator.state.auto_mode_state, "ACTIVE")

        # Trigger Emergency Stop
        res_estop = self.client.post("/api/emergency-stop")
        self.assertEqual(res_estop.status_code, 200)

        # Auto Mode must immediately enter BLOCKED
        auto_res = self.client.get("/api/auto-mode").get_json()
        self.assertEqual(auto_res["state"], "BLOCKED")
        self.assertFalse(auto_res["enabled"])
        self.assertFalse(auto_res["movement_authority"])
        self.assertEqual(simulator.state.last_command, "stop")

        # Attempting to re-enable Auto Mode while E-Stop is active must be rejected (HTTP 400)
        res_enable = self.client.post("/api/auto-mode/enable")
        self.assertEqual(res_enable.status_code, 400)
        self.assertEqual(res_enable.get_json()["status"], "rejected")

    def test_13_reset_estop_does_not_resume_auto_mode(self):
        """13. Reset E-Stop does not automatically resume Auto Mode."""
        self.client.post("/api/auto-mode/enable")
        self.client.post("/api/emergency-stop")
        self.assertEqual(simulator.state.auto_mode_state, "BLOCKED")

        # Reset Emergency Stop
        res_reset = self.client.post("/api/reset-emergency-stop")
        self.assertEqual(res_reset.status_code, 200)

        # Auto Mode must remain DISABLED/non-active; NO automatic locomotion resumes
        auto_res = self.client.get("/api/auto-mode").get_json()
        self.assertFalse(auto_res["enabled"])
        self.assertIn(auto_res["state"], ("DISABLED", "READY"))
        self.assertFalse(auto_res["movement_authority"])
        self.assertEqual(simulator.state.last_command, "stop")

    def test_14_disable_auto_mode_produces_stop(self):
        """14. Disable Auto Mode produces STOP."""
        self.client.post("/api/auto-mode/enable")
        self.assertEqual(simulator.state.auto_mode_state, "ACTIVE")

        res_disable = self.client.post("/api/auto-mode/disable")
        self.assertEqual(res_disable.status_code, 200)
        data = res_disable.get_json()
        self.assertFalse(data["enabled"])
        self.assertEqual(data["state"], "DISABLED")
        self.assertFalse(data["movement_authority"])
        self.assertEqual(simulator.state.last_command, "stop")

    def test_15_16_17_simulation_isolation_preserved(self):
        """15. Hardware remains inactive; 16. mode remains simulation; 17. hardware_active remains false."""
        res = self.client.get("/api/auto-mode").get_json()
        self.assertEqual(res["mode"], "simulation")
        self.assertFalse(res["hardware_active"])
        self.assertTrue(res["simulated"])

        status_res = self.client.get("/api/status").get_json()
        self.assertEqual(status_res["mode"], "simulation")
        self.assertFalse(status_res["hardware_active"])
        self.assertTrue(status_res["simulated"])

    def test_manual_command_blocked_when_auto_active(self):
        """Manual movement commands must be rejected while Auto Mode holds authority."""
        self.client.post("/api/auto-mode/enable")
        self.assertEqual(simulator.state.auto_mode_state, "ACTIVE")

        # Movement command should be rejected
        res = self.client.post("/api/commands", json={"command": "forward"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("auto mode is active", res.get_json().get("reason", "").lower())

        # STOP command should still be accepted
        res_stop = self.client.post("/api/commands", json={"command": "stop"})
        self.assertEqual(res_stop.status_code, 200)


class TestDestinationAutoPipeline(unittest.TestCase):
    """
    Unit tests for Destination Auto Mode (2D simulated waypoint navigation,
    route planning, obstacle avoidance, arrival detection, safety hierarchy,
    and mode isolation).
    """

    def setUp(self):
        self.client = app.test_client()
        simulator.reset_emergency_stop()
        simulator.disable_auto_mode()
        simulator.disable_destination_auto()
        simulator.dest_state.current_position = Position(0.0, 0.0)
        simulator.dest_state.heading = 0.0
        simulator.dest_state.active_scenario = "destination_nominal"
        simulator.apply_scenario("nominal")

    def test_01_destinations_list(self):
        """1. GET /api/destinations returns available simulated test destinations."""
        res = self.client.get("/api/destinations")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("destinations", data)
        d_ids = [d["id"] for d in data["destinations"]]
        self.assertIn("START_POINT", d_ids)
        self.assertIn("KITCHEN", d_ids)
        self.assertIn("LIVING_ROOM", d_ids)
        self.assertIn("CHARGING_STATION", d_ids)
        self.assertIn("BEDROOM", d_ids)

    def test_02_destination_auto_starts_idle(self):
        """2. Destination Auto starts in IDLE state with disabled authority."""
        res = self.client.get("/api/destination-auto")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertFalse(data["enabled"])
        self.assertEqual(data["state"], "IDLE")
        self.assertFalse(data["movement_authority"])
        self.assertEqual(data["current_command"], "STOP")
        self.assertEqual(data["mode"], "simulation")
        self.assertFalse(data["hardware_active"])

    def test_03_select_destination_and_plan_route(self):
        """3. Selecting destination plans deterministic waypoint route."""
        res = self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["destination"]["id"], "KITCHEN")
        self.assertEqual(data["destination"]["name"], "Kitchen")
        self.assertEqual(data["state"], "ROUTE_PLANNED")
        self.assertIsNotNone(data["route"])
        self.assertGreaterEqual(len(data["route"]["waypoints"]), 1)
        self.assertGreater(data["distance_remaining"], 0.0)

    def test_04_get_route_endpoint(self):
        """4. GET /api/destination-auto/route returns active route."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "LIVING_ROOM"})
        res = self.client.get("/api/destination-auto/route")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertIsNotNone(data["route"])
        self.assertEqual(data["destination"]["id"], "LIVING_ROOM")

    def test_05_start_navigation(self):
        """5. Starting navigation enters NAVIGATING state with movement authority."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        res = self.client.post("/api/destination-auto/start")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["enabled"])
        self.assertEqual(data["state"], "NAVIGATING")
        self.assertTrue(data["movement_authority"])
        self.assertEqual(data["safety_state"], "SAFE")

    def test_06_stop_navigation(self):
        """6. Stopping navigation halts locomotion and releases authority."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/stop")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "STOPPED")
        self.assertFalse(data["movement_authority"])
        self.assertEqual(data["current_command"], "STOP")

    def test_07_scenario_destination_nominal(self):
        """7. destination_nominal -> NAVIGATING (FORWARD toward waypoint)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_nominal"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "NAVIGATING")
        self.assertIn(data["current_command"], ("FORWARD", "TURN_LEFT", "TURN_RIGHT"))
        self.assertTrue(data["movement_authority"])
        self.assertEqual(data["safety_state"], "SAFE")

    def test_08_scenario_obstacle_front(self):
        """8. destination_obstacle_front (<30cm) -> OBSTACLE_AVOIDANCE (STOP)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_obstacle_front"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "OBSTACLE_AVOIDANCE")
        self.assertEqual(data["current_command"], "STOP")

    def test_09_scenario_obstacle_left(self):
        """9. destination_obstacle_left (<35cm) -> OBSTACLE_AVOIDANCE (TURN_RIGHT)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_obstacle_left"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "OBSTACLE_AVOIDANCE")
        self.assertEqual(data["current_command"], "TURN_RIGHT")
        self.assertTrue(data["movement_authority"])

    def test_10_scenario_obstacle_right(self):
        """10. destination_obstacle_right (<35cm) -> OBSTACLE_AVOIDANCE (TURN_LEFT)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_obstacle_right"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "OBSTACLE_AVOIDANCE")
        self.assertEqual(data["current_command"], "TURN_LEFT")
        self.assertTrue(data["movement_authority"])

    def test_11_scenario_person_center_safety_stop(self):
        """11. destination_person_center -> STOPPED (Safety STOP)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_person_center"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "STOPPED")
        self.assertEqual(data["current_command"], "STOP")
        self.assertFalse(data["movement_authority"])

    def test_12_scenario_stairs_ahead_safety_stop(self):
        """12. destination_stairs_ahead -> STOPPED (Safety STOP)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_stairs_ahead"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "STOPPED")
        self.assertEqual(data["current_command"], "STOP")
        self.assertFalse(data["movement_authority"])

    def test_13_scenario_sensor_failure_rule9(self):
        """13. destination_sensor_failure -> STOPPED (Rule 9 WAIT/STOP)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_sensor_failure"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "STOPPED")
        self.assertIn(data["current_command"], ("WAIT", "STOP"))
        self.assertFalse(data["movement_authority"])

    def test_14_scenario_arrival(self):
        """14. destination_arrival -> ARRIVED (STOP, authority released)."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_arrival"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "ARRIVED")
        self.assertEqual(data["current_command"], "STOP")
        self.assertFalse(data["movement_authority"])

    def test_15_emergency_stop_locks_destination_auto(self):
        """15. Emergency Stop locks Destination Auto to BLOCKED."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")

        # Trigger E-Stop
        res_estop = self.client.post("/api/emergency-stop")
        self.assertEqual(res_estop.status_code, 200)

        # Destination Auto must be BLOCKED with authority revoked
        dest_res = self.client.get("/api/destination-auto").get_json()
        self.assertFalse(dest_res["enabled"])
        self.assertEqual(dest_res["state"], "BLOCKED")
        self.assertFalse(dest_res["movement_authority"])
        self.assertEqual(dest_res["safety_state"], "BLOCKED")
        self.assertEqual(dest_res["current_command"], "STOP")

    def test_16_estop_reset_does_not_auto_resume(self):
        """16. Resetting Emergency Stop must NOT auto-resume Destination Navigation."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")
        self.client.post("/api/emergency-stop")

        # Reset E-Stop
        res_reset = self.client.post("/api/reset-emergency-stop")
        self.assertEqual(res_reset.status_code, 200)

        # Must remain non-navigating (STOPPED or IDLE); never auto-resume
        dest_res = self.client.get("/api/destination-auto").get_json()
        self.assertFalse(dest_res["enabled"])
        self.assertIn(dest_res["state"], ("STOPPED", "IDLE"))
        self.assertFalse(dest_res["movement_authority"])
        self.assertEqual(dest_res["current_command"], "STOP")
        self.assertEqual(simulator.state.last_command, "stop")

    def test_17_mode_isolation(self):
        """17. Free Auto and Destination Auto are mutually isolated and never run together."""
        # Enable Free Auto
        self.client.post("/api/auto-mode/enable")
        self.assertTrue(simulator.state.auto_mode_enabled)
        self.assertEqual(simulator.state.auto_mode_state, "ACTIVE")

        # Starting Destination Auto stops Free Auto
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")

        self.assertTrue(simulator.dest_state.enabled)
        self.assertEqual(simulator.dest_state.state, "NAVIGATING")
        self.assertFalse(simulator.state.auto_mode_enabled)
        self.assertNotEqual(simulator.state.auto_mode_state, "ACTIVE")

        # Enabling Free Auto stops Destination Auto
        self.client.post("/api/auto-mode/enable")
        self.assertTrue(simulator.state.auto_mode_enabled)
        self.assertEqual(simulator.state.auto_mode_state, "ACTIVE")
        self.assertFalse(simulator.dest_state.enabled)
        self.assertNotEqual(simulator.dest_state.state, "NAVIGATING")

    def test_18_manual_command_blocked_when_dest_navigating(self):
        """18. Manual movement commands rejected while Destination Navigation holds authority."""
        self.client.post("/api/destination-auto/destination", json={"destination_id": "KITCHEN"})
        self.client.post("/api/destination-auto/start")

        res = self.client.post("/api/commands", json={"command": "forward"})
        self.assertEqual(res.status_code, 400)
        self.assertIn("destination auto mode is active", res.get_json().get("reason", "").lower())

        # STOP command must be accepted
        res_stop = self.client.post("/api/commands", json={"command": "stop"})
        self.assertEqual(res_stop.status_code, 200)

    def test_19_hardware_safety_flags_preserved(self):
        """19. Hardware remains decoupled (mode='simulation', hardware_active=False)."""
        res = self.client.get("/api/destination-auto").get_json()
        self.assertEqual(res["mode"], "simulation")
        self.assertFalse(res["hardware_active"])
        self.assertTrue(res["simulated"])


class TestDestinationMapCoordinatesAndArbitraryNavigation(unittest.TestCase):
    """
    Tests for arbitrary simulated map coordinate selection, bounds validation,
    deterministic waypoint progression, arrival, obstacle safety, and destination clearing.
    """

    def setUp(self):
        self.app = app
        self.client = self.app.test_client()
        simulator.reset_emergency_stop()
        simulator.dest_state.current_position = Position(0.0, 0.0)
        simulator.dest_state.heading = 0.0
        simulator.clear_destination()
        simulator.apply_scenario("nominal")

    def tearDown(self):
        simulator.clear_destination()
        simulator.disable_auto_mode()
        simulator.reset_emergency_stop()

    def test_01_arbitrary_destination_selection(self):
        """1. Arbitrary destination coordinate selection without plan_route sets DESTINATION_SELECTED."""
        res = self.client.post("/api/destination-auto/destination", json={"x": 7.4, "y": 5.8, "plan_route": False})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["state"], "DESTINATION_SELECTED")
        self.assertEqual(data["destination"]["position"]["x"], 7.4)
        self.assertEqual(data["destination"]["position"]["y"], 5.8)
        self.assertIsNone(data["route"])

    def test_02_positive_coordinates(self):
        """2. Positive coordinate target (8.4, 6.2) with route planning generates valid waypoints."""
        res = self.client.post("/api/destination-auto/destination", json={"x": 8.4, "y": 6.2, "plan_route": True})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "ROUTE_PLANNED")
        self.assertEqual(data["destination"]["position"]["x"], 8.4)
        self.assertEqual(data["destination"]["position"]["y"], 6.2)
        self.assertIsNotNone(data["route"])
        self.assertGreaterEqual(len(data["route"]["waypoints"]), 2)
        self.assertAlmostEqual(data["route"]["total_distance"], 10.4, delta=0.5)
        self.assertAlmostEqual(data["distance_remaining"], 10.4, delta=0.5)

    def test_03_negative_coordinates(self):
        """3. Negative coordinate target (-6.5, -4.2) plans deterministic route in quadrant III."""
        res = self.client.post("/api/destination-auto/destination", json={"x": -6.5, "y": -4.2, "plan_route": True})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "ROUTE_PLANNED")
        self.assertEqual(data["destination"]["position"]["x"], -6.5)
        self.assertEqual(data["destination"]["position"]["y"], -4.2)
        self.assertIsNotNone(data["route"])
        self.assertGreaterEqual(len(data["route"]["waypoints"]), 2)
        for wp in data["route"]["waypoints"]:
            self.assertLessEqual(wp["x"], 0.0)
            self.assertLessEqual(wp["y"], 0.0)

    def test_04_destination_replacement(self):
        """4. Selecting a new destination replaces previous destination, clears route, returns to DESTINATION_SELECTED."""
        self.client.post("/api/destination-auto/destination", json={"x": 8.4, "y": 6.2, "plan_route": True})
        res = self.client.post("/api/destination-auto/destination", json={"x": -3.0, "y": 4.0, "plan_route": False})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "DESTINATION_SELECTED")
        self.assertEqual(data["destination"]["position"]["x"], -3.0)
        self.assertEqual(data["destination"]["position"]["y"], 4.0)
        self.assertIsNone(data["route"])

    def test_05_invalid_x_coordinate(self):
        """5. Invalid non-numeric or NaN/Inf X coordinate is rejected with 400 Bad Request."""
        res_str = self.client.post("/api/destination-auto/destination", json={"x": "invalid_val", "y": 5.0})
        self.assertEqual(res_str.status_code, 400)
        self.assertIn("numeric", res_str.get_json()["error"])

        res_nan = self.client.post("/api/destination-auto/destination", json={"x": float("nan"), "y": 5.0})
        self.assertEqual(res_nan.status_code, 400)
        self.assertIn("finite", res_nan.get_json()["error"])

    def test_06_invalid_y_coordinate(self):
        """6. Invalid non-numeric or NaN/Inf Y coordinate is rejected with 400 Bad Request."""
        res_str = self.client.post("/api/destination-auto/destination", json={"x": 5.0, "y": "invalid_val"})
        self.assertEqual(res_str.status_code, 400)

        res_inf = self.client.post("/api/destination-auto/destination", json={"x": 5.0, "y": float("inf")})
        self.assertEqual(res_inf.status_code, 400)
        self.assertIn("finite", res_inf.get_json()["error"])

    def test_07_out_of_bounds_destination(self):
        """7. Coordinates exceeding 1.5 km circular operating radius (>1500.0m) rejected with 400."""
        res_pos = self.client.post("/api/destination-auto/destination", json={"x": 1501.0, "y": 0.0})
        self.assertEqual(res_pos.status_code, 400)
        self.assertIn("outside the 1.5 km simulation radius", res_pos.get_json()["error"])

        res_neg = self.client.post("/api/destination-auto/destination", json={"x": 0.0, "y": -1501.0})
        self.assertEqual(res_neg.status_code, 400)
        self.assertIn("outside the 1.5 km simulation radius", res_neg.get_json()["error"])

        res_diag = self.client.post("/api/destination-auto/destination", json={"x": 1200.0, "y": 1200.0})
        self.assertEqual(res_diag.status_code, 400)
        self.assertIn("outside the 1.5 km simulation radius", res_diag.get_json()["error"])

    def test_08_route_planning(self):
        """8. Route planning endpoint computes deterministic intermediate waypoints for custom target."""
        self.client.post("/api/destination-auto/destination", json={"x": 6.0, "y": 8.0, "plan_route": False})
        res = self.client.get("/api/destination-auto/route")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNotNone(data["route"])
        self.assertAlmostEqual(data["total_distance"], 10.0, delta=0.5)

    def test_09_navigation_start(self):
        """9. Starting navigation acquires movement authority and enters NAVIGATING state."""
        self.client.post("/api/destination-auto/destination", json={"x": 5.0, "y": 5.0, "plan_route": True})
        res = self.client.post("/api/destination-auto/start")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "NAVIGATING")
        self.assertTrue(data["movement_authority"])
        self.assertIn(data["current_command"], ("FORWARD", "TURN_LEFT", "TURN_RIGHT"))

    def test_10_waypoint_progression(self):
        """10. Waypoint progression advances waypoint index as robot moves along route."""
        self.client.post("/api/destination-auto/destination", json={"x": 4.0, "y": 0.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        for _ in range(10):
            simulator._step_destination_auto()
        self.assertGreater(simulator.dest_state.current_position.x, 0.0)

    def test_11_arrival_at_arbitrary_destination(self):
        """11. Arrival: when position reaches within arrival threshold, transitions to ARRIVED and halts."""
        self.client.post("/api/destination-auto/destination", json={"x": 5.0, "y": 5.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        simulator.dest_state.current_position = Position(5.1, 4.9)
        simulator._step_destination_auto()
        self.assertEqual(simulator.dest_state.state, "ARRIVED")
        self.assertEqual(simulator.dest_state.current_command, "STOP")
        self.assertFalse(simulator.dest_state.movement_authority)

    def test_12_obstacle_handling(self):
        """12. Obstacle during navigation engages OBSTACLE_AVOIDANCE and halts or steers safely."""
        self.client.post("/api/destination-auto/destination", json={"x": 6.0, "y": 0.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_obstacle_front"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "OBSTACLE_AVOIDANCE")
        self.assertEqual(data["current_command"], "STOP")

    def test_13_person_safety_stop(self):
        """13. Person in path halts locomotion immediately with safety STOP."""
        self.client.post("/api/destination-auto/destination", json={"x": 6.0, "y": 0.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_person_center"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "STOPPED")
        self.assertEqual(data["current_command"], "STOP")
        self.assertFalse(data["movement_authority"])

    def test_14_stairs_safety_stop(self):
        """14. Stairs detection halts locomotion with safety STOP."""
        self.client.post("/api/destination-auto/destination", json={"x": 6.0, "y": 0.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_stairs_ahead"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "STOPPED")
        self.assertEqual(data["current_command"], "STOP")
        self.assertFalse(data["movement_authority"])

    def test_15_sensor_failure_rule9(self):
        """15. Sensor fault prevents blind progression, yields WAIT or STOP (Rule 9)."""
        self.client.post("/api/destination-auto/destination", json={"x": 6.0, "y": 0.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/destination-auto/scenario", json={"scenario": "destination_sensor_failure"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "STOPPED")
        self.assertIn(data["current_command"], ("WAIT", "STOP"))
        self.assertFalse(data["movement_authority"])

    def test_16_emergency_stop_halts_and_blocks(self):
        """16. Emergency Stop overrides navigation immediately to BLOCKED."""
        self.client.post("/api/destination-auto/destination", json={"x": 6.0, "y": 0.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        res = self.client.post("/api/emergency-stop")
        self.assertEqual(res.status_code, 200)
        dest_res = self.client.get("/api/destination-auto").get_json()
        self.assertEqual(dest_res["state"], "BLOCKED")
        self.assertFalse(dest_res["movement_authority"])

    def test_17_estop_reset_does_not_resume(self):
        """17. Resetting E-Stop leaves navigation stopped without resuming movement."""
        self.client.post("/api/destination-auto/destination", json={"x": 6.0, "y": 0.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        self.client.post("/api/emergency-stop")
        self.client.post("/api/reset-emergency-stop")
        dest_res = self.client.get("/api/destination-auto").get_json()
        self.assertNotEqual(dest_res["state"], "NAVIGATING")
        self.assertFalse(dest_res["movement_authority"])
        self.assertEqual(dest_res["current_command"], "STOP")

    def test_18_authority_isolation_and_clear_destination(self):
        """18. Mode authority isolation and CLEAR DESTINATION endpoint behavior."""
        self.client.post("/api/destination-auto/destination", json={"x": 5.0, "y": 5.0, "plan_route": True})
        clear_res = self.client.post("/api/destination-auto/clear")
        self.assertEqual(clear_res.status_code, 200)
        data = clear_res.get_json()
        self.assertEqual(data["state"], "IDLE")
        self.assertIsNone(data["destination"])
        self.assertIsNone(data["route"])
        self.assertFalse(data["movement_authority"])

        # 18b: Mode Isolation: When Free Auto is enabled, Destination Navigation is stopped
        self.client.post("/api/destination-auto/destination", json={"x": 5.0, "y": 5.0, "plan_route": True})
        self.client.post("/api/destination-auto/start")
        self.assertEqual(simulator.dest_state.state, "NAVIGATING")
        self.assertTrue(simulator.dest_state.movement_authority)

        self.client.post("/api/auto-mode/enable")
        self.assertTrue(simulator.state.auto_mode_enabled)
        self.assertEqual(simulator.state.auto_mode_state, "ACTIVE")
        self.assertNotEqual(simulator.dest_state.state, "NAVIGATING")
        self.assertFalse(simulator.dest_state.movement_authority)

        # 18c: Manual movement commands rejected while autonomous mode holds authority
        man_cmd = self.client.post("/api/commands", json={"command": "forward"})
        self.assertEqual(man_cmd.status_code, 400)

    def test_19_exact_screenshot_regression_coordinate_conversion(self):
        """
        19. CRITICAL REGRESSION TEST:
        Symptom:
          Current simulated position: Lat 12.971587, Lon 77.594784 (Origin)
          Selected destination:       Lat 12.971420, Lon 77.594784 (South of origin)
        Problem:
          Previous UI bug clamped this point to X = 20.0 m, Y = -20.0 m.
        Correction:
          Longitude diff is 0.0 -> X must be ~0.0 m (NOT +20.0 m).
          Latitude diff is -0.000167 deg -> Y must be ~ -18.6 m (NOT -20.0 m).
          Total distance must be ~18.6 m (NOT 28.3 m).
        """
        origin_lat = 12.971587
        origin_lon = 77.594784
        clicked_lat = 12.971420
        clicked_lon = 77.594784

        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(origin_lat))

        x_meters = (clicked_lon - origin_lon) * meters_per_deg_lon
        y_meters = (clicked_lat - origin_lat) * meters_per_deg_lat
        dist_meters = math.hypot(x_meters, y_meters)

        # Coordinate assertions
        self.assertAlmostEqual(x_meters, 0.0, places=2)
        self.assertAlmostEqual(y_meters, -18.59, delta=0.1)
        self.assertAlmostEqual(dist_meters, 18.59, delta=0.1)

        # Verify it is NEVER clamped to ±20.0
        self.assertNotEqual(round(x_meters, 1), 20.0)
        self.assertNotEqual(round(y_meters, 1), -20.0)

        # Submit to backend destination endpoint
        res = self.client.post(
            "/api/destination-auto/destination",
            json={"x": round(x_meters, 2), "y": round(y_meters, 2), "plan_route": True},
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["state"], "ROUTE_PLANNED")
        self.assertAlmostEqual(data["destination"]["position"]["x"], 0.0, places=2)
        self.assertAlmostEqual(data["destination"]["position"]["y"], -18.59, delta=0.1)
        self.assertAlmostEqual(data["distance_remaining"], 18.59, delta=0.2)

    def test_20_map_click_directional_projections(self):
        """
        20. Deterministic directional map-click tests:
        A: Origin (X ≈ 0, Y ≈ 0)
        B: North  (X ≈ 0, Y > 0)
        C: South  (X ≈ 0, Y < 0)
        D: East   (X > 0, Y ≈ 0)
        E: West   (X < 0, Y ≈ 0)
        F: Northeast (X > 0, Y > 0)
        G: Southwest (X < 0, Y < 0)
        """
        origin_lat = 12.971587
        origin_lon = 77.594784
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(origin_lat))

        cases = [
            ("A_origin", origin_lat, origin_lon, 0.0, 0.0),
            ("B_north", origin_lat + 10.0 / meters_per_deg_lat, origin_lon, 0.0, 10.0),
            ("C_south", origin_lat - 10.0 / meters_per_deg_lat, origin_lon, 0.0, -10.0),
            ("D_east", origin_lat, origin_lon + 10.0 / meters_per_deg_lon, 10.0, 0.0),
            ("E_west", origin_lat, origin_lon - 10.0 / meters_per_deg_lon, -10.0, 0.0),
            ("F_northeast", origin_lat + 10.0 / meters_per_deg_lat, origin_lon + 10.0 / meters_per_deg_lon, 10.0, 10.0),
            ("G_southwest", origin_lat - 10.0 / meters_per_deg_lat, origin_lon - 10.0 / meters_per_deg_lon, -10.0, -10.0),
        ]

        for label, lat, lon, exp_x, exp_y in cases:
            with self.subTest(case=label):
                x = (lon - origin_lon) * meters_per_deg_lon
                y = (lat - origin_lat) * meters_per_deg_lat

                self.assertAlmostEqual(x, exp_x, delta=0.05, msg=f"{label} X mismatch")
                self.assertAlmostEqual(y, exp_y, delta=0.05, msg=f"{label} Y mismatch")

                res = self.client.post(
                    "/api/destination-auto/destination",
                    json={"x": round(x, 2), "y": round(y, 2), "plan_route": False},
                )
                self.assertEqual(res.status_code, 200)
                data = res.get_json()
                self.assertEqual(data["state"], "DESTINATION_SELECTED")
                self.assertAlmostEqual(data["destination"]["position"]["x"], exp_x, delta=0.05)
                self.assertAlmostEqual(data["destination"]["position"]["y"], exp_y, delta=0.05)

    def test_21_out_of_bounds_rejection_no_clamping(self):
        """
        21. Points outside 1.5 km circular boundary (sqrt(x^2 + y^2) > 1500m)
        are rejected with 400 Bad Request, NEVER silently clamped.
        """
        out_of_bounds_cases = [
            (1501.0, 0.0),
            (0.0, -1501.0),
            (1200.0, 1200.0),   # dist ≈ 1697m
            (-1100.0, -1100.0), # dist ≈ 1555m
        ]
        for x, y in out_of_bounds_cases:
            with self.subTest(x=x, y=y):
                res = self.client.post("/api/destination-auto/destination", json={"x": x, "y": y})
                self.assertEqual(res.status_code, 400)
                data = res.get_json()
                self.assertIn("outside the 1.5 km simulation radius", data.get("error", ""))

    def test_22_clear_destination_preserves_current_robot_position(self):
        """
        22. CLEAR DESTINATION must clear destination, route, and waypoints,
        reset state to IDLE, but MUST NOT reset the robot's current position to origin.
        """
        # Place robot away from origin
        simulator.dest_state.current_position = Position(3.5, 4.2)
        simulator.dest_state.heading = 45.0

        # Select a destination and plan route
        self.client.post("/api/destination-auto/destination", json={"x": 8.0, "y": 8.0, "plan_route": True})
        self.assertEqual(simulator.dest_state.state, "ROUTE_PLANNED")

        # Clear destination
        clear_res = self.client.post("/api/destination-auto/clear")
        self.assertEqual(clear_res.status_code, 200)
        data = clear_res.get_json()
        self.assertEqual(data["state"], "IDLE")
        self.assertIsNone(data["destination"])
        self.assertIsNone(data["route"])
        self.assertFalse(data["movement_authority"])

        # Crucial check: Robot remains at (3.5, 4.2), NOT reset to (0, 0)
        self.assertEqual(simulator.dest_state.current_position.x, 3.5)
        self.assertEqual(simulator.dest_state.current_position.y, 4.2)
        self.assertEqual(data["current_position"]["x"], 3.5)
        self.assertEqual(data["current_position"]["y"], 4.2)

    def test_23_heading_conventions_and_conversions(self):
        """
        23. Heading conversion verification:
        Simulation internal: 0° = +X (East), 90° = +Y (North), 180° = -X (West), 270° = -Y (South)
        Compass heading:     0° = North, 90° = East, 180° = South, 270° = West
        Formula: compass = (450 - sim) % 360, sim = (450 - compass) % 360
        """
        def sim_to_compass(sim):
            return (450 - (sim % 360)) % 360

        def compass_to_sim(compass):
            return (450 - (compass % 360)) % 360

        self.assertEqual(sim_to_compass(0.0), 90.0)    # Sim East -> Compass East
        self.assertEqual(sim_to_compass(90.0), 0.0)    # Sim North -> Compass North
        self.assertEqual(sim_to_compass(180.0), 270.0) # Sim West -> Compass West
        self.assertEqual(sim_to_compass(270.0), 180.0) # Sim South -> Compass South
        self.assertEqual(sim_to_compass(45.0), 45.0)   # Sim NE -> Compass NE

        # Roundtrip
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            self.assertEqual(compass_to_sim(sim_to_compass(angle)), angle)

    def test_24_reverse_conversion_local_to_geographic(self):
        """
        24. Verifies round-trip geographic <-> local metric conversion accuracy:
        (lat, lon) -> (x, y) -> (lat, lon)
        """
        origin_lat = 12.971587
        origin_lon = 77.594784
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(origin_lat))

        test_points = [
            (12.971587, 77.594784),
            (12.971420, 77.594784),
            (12.971650, 77.594850),
            (12.971500, 77.594700),
        ]

        for orig_lat, orig_lon in test_points:
            with self.subTest(lat=orig_lat, lon=orig_lon):
                x = (orig_lon - origin_lon) * meters_per_deg_lon
                y = (orig_lat - origin_lat) * meters_per_deg_lat

                calc_lat = origin_lat + y / meters_per_deg_lat
                calc_lon = origin_lon + x / meters_per_deg_lon

                self.assertAlmostEqual(orig_lat, calc_lat, places=6)
                self.assertAlmostEqual(orig_lon, calc_lon, places=6)

    def test_25_expand_1_5km_radius_valid_and_invalid_examples(self):
        """
        25. Validates 1.5 km circular operating boundary (Section 19 tests 1-9):
        Valid (<=1500m):
          1. Origin: (0, 0)
          2. 500m: (500, 0)
          3. 1000m: (1000, 0)
          4. 1500m: (1500, 0)
          5. -1500m: (-1500, 0)
          6. 1500m North: (0, 1500)
          7. 1500m Southwest: (-1060.66, -1060.66) [dist ≈ 1500m]
          - Additional: (1000, 1000) [dist ≈ 1414.21m]
          - Additional: (1200, 600) [dist ≈ 1341.64m]
        Invalid (>1500m):
          8. 1501m: (1501, 0)
          - 1501m South: (0, -1501)
          9. 1200m/1200m: (1200, 1200) [dist ≈ 1697.06m]
        """
        valid_cases = [
            ("origin", 0.0, 0.0),
            ("500m_east", 500.0, 0.0),
            ("1000m_east", 1000.0, 0.0),
            ("1500m_east", 1500.0, 0.0),
            ("1500m_west", -1500.0, 0.0),
            ("1500m_north", 0.0, 1500.0),
            ("1500m_south", 0.0, -1500.0),
            ("1500m_southwest", -1060.66, -1060.66),
            ("1000_1000_diagonal", 1000.0, 1000.0),
            ("1200_600_diagonal", 1200.0, 600.0),
        ]

        for label, x, y in valid_cases:
            with self.subTest(valid_case=label, x=x, y=y):
                res = self.client.post("/api/destination-auto/destination", json={"x": x, "y": y, "plan_route": False})
                self.assertEqual(res.status_code, 200, f"Expected 200 for valid point {label} ({x}, {y})")
                data = res.get_json()
                self.assertEqual(data["state"], "DESTINATION_SELECTED")
                self.assertAlmostEqual(data["destination"]["position"]["x"], x, places=2)
                self.assertAlmostEqual(data["destination"]["position"]["y"], y, places=2)

        invalid_cases = [
            ("1501m_east", 1501.0, 0.0),
            ("1501m_south", 0.0, -1501.0),
            ("1200_1200_diagonal", 1200.0, 1200.0),
            ("1100_1100_diagonal", -1100.0, -1100.0),
        ]

        for label, x, y in invalid_cases:
            with self.subTest(invalid_case=label, x=x, y=y):
                res = self.client.post("/api/destination-auto/destination", json={"x": x, "y": y, "plan_route": False})
                self.assertEqual(res.status_code, 400, f"Expected 400 for out-of-bounds point {label} ({x}, {y})")
                data = res.get_json()
                self.assertIn("outside the 1.5 km simulation radius", data.get("error", ""))

    def test_26_long_route_deterministic_waypoint_planning(self):
        """
        26. Long-route waypoint planning (Sections 11 & 12):
        - Spacing between 100m and 300m for long routes (>20m).
        - 1500m route produces 10 waypoints (~150m spacing).
        - 500m route produces 4 waypoints (~125m spacing).
        - Does NOT create excessive waypoints.
        - Deterministic waypoint progression and total distance accuracy.
        """
        # Test 1500m boundary route
        res_1500 = self.client.post(
            "/api/destination-auto/destination",
            json={"x": 1500.0, "y": 0.0, "plan_route": True},
        )
        self.assertEqual(res_1500.status_code, 200)
        data_1500 = res_1500.get_json()
        self.assertEqual(data_1500["state"], "ROUTE_PLANNED")
        route_1500 = data_1500["route"]
        self.assertIsNotNone(route_1500)
        self.assertAlmostEqual(route_1500["total_distance"], 1500.0, delta=0.5)

        wps_1500 = route_1500["waypoints"]
        self.assertGreaterEqual(len(wps_1500), 4)
        self.assertLessEqual(len(wps_1500), 15)
        # Check waypoint spacing (100m - 300m)
        for i in range(len(wps_1500) - 1):
            wp_a = wps_1500[i]
            wp_b = wps_1500[i + 1]
            spacing = math.hypot(wp_b["x"] - wp_a["x"], wp_b["y"] - wp_a["y"])
            self.assertGreaterEqual(spacing, 80.0, f"Waypoint spacing too tight: {spacing}m")
            self.assertLessEqual(spacing, 320.0, f"Waypoint spacing too large: {spacing}m")

        # Last waypoint must reach destination (1500, 0)
        self.assertAlmostEqual(wps_1500[-1]["x"], 1500.0, places=2)
        self.assertAlmostEqual(wps_1500[-1]["y"], 0.0, places=2)

        # Test 500m route
        res_500 = self.client.post(
            "/api/destination-auto/destination",
            json={"x": 0.0, "y": 500.0, "plan_route": True},
        )
        self.assertEqual(res_500.status_code, 200)
        data_500 = res_500.get_json()
        self.assertAlmostEqual(data_500["route"]["total_distance"], 500.0, delta=0.5)
        self.assertGreaterEqual(len(data_500["route"]["waypoints"]), 3)
        self.assertLessEqual(len(data_500["route"]["waypoints"]), 6)

    def test_27_long_route_navigation_progression_and_arrival(self):
        """
        27. Navigation progression and arrival detection for long-distance 1.5 km routes:
        - Acquires authority upon start.
        - Distance remaining decreases as robot advances.
        - Transitions to ARRIVED and halts when destination reached within 0.5m.
        """
        # Start at origin
        simulator.dest_state.current_position = Position(0.0, 0.0)
        self.client.post(
            "/api/destination-auto/destination",
            json={"x": 1500.0, "y": 0.0, "plan_route": True},
        )
        start_res = self.client.post("/api/destination-auto/start")
        self.assertEqual(start_res.status_code, 200)
        data = start_res.get_json()
        self.assertEqual(data["state"], "NAVIGATING")
        self.assertTrue(data["movement_authority"])
        self.assertAlmostEqual(data["distance_remaining"], 1500.0, delta=1.0)

        # Advance simulator steps
        for _ in range(5):
            simulator._step_destination_auto()

        self.assertGreater(simulator.dest_state.current_position.x, 0.0)
        updated_status = self.client.get("/api/destination-auto").get_json()
        self.assertLess(updated_status["distance_remaining"], 1500.0)

        # Simulate arrival at 1500m destination
        simulator.dest_state.current_position = Position(1499.8, 0.0)
        simulator._step_destination_auto()

        arrived_status = self.client.get("/api/destination-auto").get_json()
        self.assertEqual(arrived_status["state"], "ARRIVED")
        self.assertEqual(arrived_status["current_command"], "STOP")
        self.assertFalse(arrived_status["movement_authority"])


if __name__ == "__main__":
    unittest.main()
