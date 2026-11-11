"""
Unit Tests for SpotMicro Navigation Decision Engine (Stage 1)
=============================================================
Tests all high-level decision rules, sensor validation, safety priorities,
hysteresis dampening, dead-man policy, and the GaitController integration bridge.

All tests run strictly using mocked inputs and do NOT touch physical hardware
(no GPIO, I2C, PCA9685, or physical servos).
"""

import math
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.navigation import (
    BoundingBox,
    Detection,
    IMUReading,
    NavCommand,
    NavigationConfig,
    NavigationDecision,
    NavigationDecisionEngine,
    NavigationGaitBridge,
    NavigationInput,
    UltrasonicReading,
    UltrasonicSensors,
    YOLOPerceptionAdapter,
)
from app.gait_controller import GaitController


class TestBoundingBoxAndValidation(unittest.TestCase):
    """Tests for BoundingBox coordinate constraints and spatial queries."""

    def test_valid_bounding_box(self):
        bbox = BoundingBox(x_min=0.2, y_min=0.1, x_max=0.6, y_max=0.8)
        self.assertAlmostEqual(bbox.width, 0.4)
        self.assertAlmostEqual(bbox.height, 0.7)
        self.assertAlmostEqual(bbox.center_x, 0.4)
        self.assertAlmostEqual(bbox.center_y, 0.45)
        self.assertAlmostEqual(bbox.area, 0.28)

    def test_invalid_bounding_box_reversed_coordinates(self):
        with self.assertRaises(ValueError):
            BoundingBox(x_min=0.7, y_min=0.1, x_max=0.3, y_max=0.8)  # x_min >= x_max

        with self.assertRaises(ValueError):
            BoundingBox(x_min=0.2, y_min=0.8, x_max=0.6, y_max=0.4)  # y_min >= y_max

    def test_invalid_bounding_box_out_of_bounds(self):
        with self.assertRaises(ValueError):
            BoundingBox(x_min=-0.1, y_min=0.0, x_max=0.5, y_max=0.5)

        with self.assertRaises(ValueError):
            BoundingBox(x_min=0.0, y_min=0.0, x_max=1.2, y_max=0.5)

    def test_invalid_bounding_box_nan_inf(self):
        with self.assertRaises(ValueError):
            BoundingBox(x_min=float("nan"), y_min=0.0, x_max=0.5, y_max=0.5)

        with self.assertRaises(ValueError):
            BoundingBox(x_min=0.0, y_min=0.0, x_max=float("inf"), y_max=0.5)

    def test_horizontal_region_classification(self):
        left_box = BoundingBox(x_min=0.05, y_min=0.2, x_max=0.25, y_max=0.8)
        self.assertEqual(left_box.horizontal_region(), "LEFT")

        center_box = BoundingBox(x_min=0.35, y_min=0.2, x_max=0.65, y_max=0.8)
        self.assertEqual(center_box.horizontal_region(), "CENTER")

        right_box = BoundingBox(x_min=0.75, y_min=0.2, x_max=0.95, y_max=0.8)
        self.assertEqual(right_box.horizontal_region(), "RIGHT")


class TestUltrasonicSensorReadings(unittest.TestCase):
    """Tests for HC-SR04 reading validation and range checks."""

    def test_valid_ultrasonic_reading(self):
        reading = UltrasonicReading(distance_cm=45.0, timestamp=time.time())
        self.assertTrue(reading.is_valid)
        self.assertEqual(reading.distance_cm, 45.0)

    def test_out_of_range_ultrasonic_readings(self):
        # Under 2 cm
        too_close = UltrasonicReading(distance_cm=1.0)
        self.assertFalse(too_close.is_valid)

        # Over 400 cm
        too_far = UltrasonicReading(distance_cm=500.0)
        self.assertFalse(too_far.is_valid)

        # Negative
        negative = UltrasonicReading(distance_cm=-10.0)
        self.assertFalse(negative.is_valid)

        # NaN
        nan_reading = UltrasonicReading(distance_cm=float("nan"))
        self.assertFalse(nan_reading.is_valid)


class TestIMUValidation(unittest.TestCase):
    """Tests for BNO055 IMU tilt limits and dynamic stability checks."""

    def test_safe_imu_reading(self):
        now = time.time()
        imu = IMUReading(roll=3.5, pitch=-2.1, yaw=180.0, is_stable=True, timestamp=now)
        safe, reason = imu.is_safe(max_roll_deg=20.0, max_pitch_deg=20.0, max_age_s=0.5, current_time=now)
        self.assertTrue(safe)
        self.assertIn("stable", reason.lower())

    def test_unsafe_roll_tilt(self):
        now = time.time()
        imu = IMUReading(roll=25.0, pitch=2.0, is_stable=True, timestamp=now)
        safe, reason = imu.is_safe(max_roll_deg=20.0, max_pitch_deg=20.0, max_age_s=0.5, current_time=now)
        self.assertFalse(safe)
        self.assertIn("roll", reason.lower())

    def test_unsafe_pitch_tilt(self):
        now = time.time()
        imu = IMUReading(roll=1.0, pitch=-24.5, is_stable=True, timestamp=now)
        safe, reason = imu.is_safe(max_roll_deg=20.0, max_pitch_deg=20.0, max_age_s=0.5, current_time=now)
        self.assertFalse(safe)
        self.assertIn("pitch", reason.lower())

    def test_unstable_imu_status(self):
        now = time.time()
        imu = IMUReading(roll=0.0, pitch=0.0, is_stable=False, timestamp=now)
        safe, reason = imu.is_safe(max_roll_deg=20.0, max_pitch_deg=20.0, max_age_s=0.5, current_time=now)
        self.assertFalse(safe)
        self.assertIn("instability", reason.lower())

    def test_stale_imu_reading(self):
        now = time.time()
        imu = IMUReading(roll=0.0, pitch=0.0, is_stable=True, timestamp=now - 2.0)
        safe, reason = imu.is_safe(max_roll_deg=20.0, max_pitch_deg=20.0, max_age_s=0.5, current_time=now)
        self.assertFalse(safe)
        self.assertIn("stale", reason.lower())


class TestNavigationDecisionEngine(unittest.TestCase):
    """Comprehensive functional tests for the NavigationDecisionEngine."""

    def setUp(self):
        # Configure engine with zero stop-hold frames and 1-frame switch for deterministic single-cycle testing
        self.config = NavigationConfig(
            front_stop_distance_cm=30.0,
            front_caution_distance_cm=65.0,
            side_clearance_distance_cm=35.0,
            max_sensor_age_s=0.5,
            deadman_timeout_s=1.0,
            consecutive_frames_required=1,
            stop_hold_frames=0,
        )
        self.engine = NavigationDecisionEngine(self.config)

    def _make_sensors(self, front=120.0, left=70.0, right=70.0, rear=100.0, timestamp=None):
        ts = timestamp if timestamp is not None else time.time()
        return UltrasonicSensors(
            front=UltrasonicReading(distance_cm=front, timestamp=ts) if front is not None else None,
            left=UltrasonicReading(distance_cm=left, timestamp=ts) if left is not None else None,
            right=UltrasonicReading(distance_cm=right, timestamp=ts) if right is not None else None,
            rear=UltrasonicReading(distance_cm=rear, timestamp=ts) if rear is not None else None,
        )

    # 1. Person blocking the center path
    def test_person_blocking_center_path(self):
        now = time.time()
        person_bbox = BoundingBox(x_min=0.35, y_min=0.2, x_max=0.65, y_max=0.9)
        detections = [Detection(class_name="person", confidence=0.85, bbox=person_bbox)]
        nav_input = NavigationInput(
            detections=detections,
            ultrasonic=self._make_sensors(front=150.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("person", decision.reason.lower())
        self.assertTrue(decision.telemetry.get("emergency_stop"))

    # 2. Obstacle blocking the center path
    def test_obstacle_blocking_center_path(self):
        now = time.time()
        obs_bbox = BoundingBox(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.8)
        detections = [Detection(class_name="obstacle", confidence=0.78, bbox=obs_bbox)]
        nav_input = NavigationInput(
            detections=detections,
            ultrasonic=self._make_sensors(front=50.0, left=80.0, right=80.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.AVOID_OBSTACLE)
        self.assertIn("obstacle", decision.reason.lower())

    # 3. Blocked left side -> Turn right
    def test_blocked_left_side(self):
        now = time.time()
        # Front is caution-blocked (50 cm), left is blocked (20 cm), right is clear (75 cm)
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=self._make_sensors(front=50.0, left=20.0, right=75.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.AVOID_OBSTACLE)
        self.assertEqual(decision.sub_action, "TURN_RIGHT")
        self.assertIn("right", decision.reason.lower())

    # 4. Blocked right side -> Turn left
    def test_blocked_right_side(self):
        now = time.time()
        # Front is caution-blocked (50 cm), left is clear (75 cm), right is blocked (20 cm)
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=self._make_sensors(front=50.0, left=75.0, right=20.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.AVOID_OBSTACLE)
        self.assertEqual(decision.sub_action, "TURN_LEFT")
        self.assertIn("left", decision.reason.lower())

    # 5. Both sides blocked -> STOP
    def test_both_sides_blocked(self):
        now = time.time()
        # Front is caution-blocked, and both lateral sensors indicate obstacle close (< 35 cm)
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=self._make_sensors(front=45.0, left=20.0, right=22.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("both lateral corridors unavailable", decision.reason.lower())

    # 6. No detections and clear path -> FORWARD
    def test_no_detections_clear_path(self):
        now = time.time()
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=self._make_sensors(front=120.0, left=60.0, right=60.0, timestamp=now),
            imu=IMUReading(roll=1.0, pitch=1.0, is_stable=True, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.FORWARD)
        self.assertIn("path verified clear", decision.reason.lower())

    # 7. Low-confidence detections filtered out
    def test_low_confidence_detections(self):
        now = time.time()
        # Person in center with very low confidence (0.15 < threshold 0.40)
        person_bbox = BoundingBox(x_min=0.35, y_min=0.2, x_max=0.65, y_max=0.9)
        detections = [Detection(class_name="person", confidence=0.15, bbox=person_bbox)]
        nav_input = NavigationInput(
            detections=detections,
            ultrasonic=self._make_sensors(front=120.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        # Should be filtered out and proceed FORWARD
        self.assertEqual(decision.command, NavCommand.FORWARD)

    # 8. Missing or stale ultrasonic data
    def test_missing_front_ultrasonic_data(self):
        now = time.time()
        # Missing front ultrasonic sensor
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=self._make_sensors(front=None, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        # Missing front sensor must NOT assume path is clear (Rule 9)
        self.assertEqual(decision.command, NavCommand.WAIT)
        self.assertIn("cannot verify path clearance", decision.reason.lower())

    def test_stale_front_ultrasonic_data(self):
        now = time.time()
        # Sensor data timestamp is 2.0s old (exceeds max_sensor_age_s 0.5s)
        stale_sensors = self._make_sensors(front=150.0, timestamp=now - 2.0)
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=stale_sensors,
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.WAIT)
        self.assertIn("stale", decision.reason.lower())

    # 9. Invalid ultrasonic readings
    def test_invalid_ultrasonic_readings(self):
        now = time.time()
        # Out-of-bounds negative reading
        negative_reading = UltrasonicReading(distance_cm=-5.0, timestamp=now)
        sensors = UltrasonicSensors(front=negative_reading)
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=sensors,
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.WAIT)
        self.assertIn("invalid or out-of-range", decision.reason.lower())

    # 10. Unsafe IMU tilt -> Emergency STOP
    def test_unsafe_imu_tilt_emergency_stop(self):
        now = time.time()
        # Excessive roll tilt (28 degrees > max 20 degrees)
        imu = IMUReading(roll=28.0, pitch=2.0, is_stable=True, timestamp=now)
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=self._make_sensors(front=150.0, timestamp=now),
            imu=imu,
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("imu safety alert", decision.reason.lower())
        self.assertIn("roll", decision.reason.lower())

    # 11. Emergency STOP priority over clear camera path
    def test_emergency_stop_priority_ultrasonic(self):
        now = time.time()
        # Camera sees completely clear path, but front ultrasonic reports object at 18 cm
        nav_input = NavigationInput(
            detections=[],
            ultrasonic=self._make_sensors(front=18.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("critically close", decision.reason.lower())

    # 12. Repeated inputs and command stability (hysteresis)
    def test_hysteresis_and_command_stability(self):
        # Configure engine requiring 3 consecutive frames to switch movement command
        config = NavigationConfig(
            consecutive_frames_required=3,
            stop_hold_frames=0,
        )
        engine = NavigationDecisionEngine(config)
        now = 1000.0

        # Prime with 3 frames of FORWARD
        for i in range(3):
            inp = NavigationInput(detections=[], ultrasonic=self._make_sensors(front=120.0, timestamp=now + i*0.1), timestamp=now + i*0.1)
            dec = engine.decide(inp)
        self.assertEqual(dec.command, NavCommand.FORWARD)

        # Single frame candidate AVOID_OBSTACLE: should be damped and stay FORWARD
        obs_bbox = BoundingBox(x_min=0.3, y_min=0.3, x_max=0.7, y_max=0.8)
        nav_input_obs = NavigationInput(
            detections=[Detection(class_name="obstacle", confidence=0.8, bbox=obs_bbox)],
            ultrasonic=self._make_sensors(front=50.0, left=70.0, right=20.0, timestamp=now + 0.4),
            timestamp=now + 0.4
        )
        dec1 = engine.decide(nav_input_obs)
        self.assertEqual(dec1.command, NavCommand.FORWARD)
        self.assertTrue(dec1.telemetry.get("filter_applied"))

        # Second consecutive frame of AVOID_OBSTACLE: still damped (2/3)
        nav_input_obs.timestamp = now + 0.5
        dec2 = engine.decide(nav_input_obs)
        self.assertEqual(dec2.command, NavCommand.FORWARD)

        # Third consecutive frame of AVOID_OBSTACLE: now confirmed (3/3)
        nav_input_obs.timestamp = now + 0.6
        dec3 = engine.decide(nav_input_obs)
        self.assertEqual(dec3.command, NavCommand.AVOID_OBSTACLE)

    # 13. Dead-man switch / pipeline stall
    def test_deadman_timeout(self):
        now = 1000.0
        # Normal cycle
        inp1 = NavigationInput(detections=[], ultrasonic=self._make_sensors(front=120.0, timestamp=now), timestamp=now)
        self.engine.decide(inp1)

        # Next cycle arrives 2.5 seconds later (> 1.0s deadman timeout)
        inp2 = NavigationInput(detections=[], ultrasonic=self._make_sensors(front=120.0, timestamp=now + 2.5), timestamp=now + 2.5)
        decision = self.engine.decide(inp2)
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("dead-man timeout", decision.reason.lower())

    # 14. Stair safety constraint (automatic stairs disabled)
    def test_stairs_detection_safety_stop(self):
        now = time.time()
        stair_bbox = BoundingBox(x_min=0.3, y_min=0.2, x_max=0.7, y_max=0.8)
        detections = [Detection(class_name="stairs", confidence=0.65, bbox=stair_bbox)]
        nav_input = NavigationInput(
            detections=detections,
            ultrasonic=self._make_sensors(front=150.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("stairs detected ahead", decision.reason.lower())
        self.assertIn("automatic stair climbing is disabled", decision.reason.lower())

    # 15. Door detection: not assumed automatically traversable
    def test_door_detection_caution(self):
        now = time.time()
        door_bbox = BoundingBox(x_min=0.3, y_min=0.2, x_max=0.7, y_max=0.8)
        detections = [Detection(class_name="door", confidence=0.75, bbox=door_bbox)]
        # Ultrasonic clearance is ambiguous / close
        nav_input = NavigationInput(
            detections=detections,
            ultrasonic=self._make_sensors(front=50.0, left=70.0, right=20.0, timestamp=now),
            timestamp=now
        )
        decision = self.engine.decide(nav_input)
        # Does NOT drive forward blindly into unverified door
        self.assertNotEqual(decision.command, NavCommand.FORWARD)


class TestYOLOPerceptionAdapter(unittest.TestCase):
    """Tests for YOLO output adapter parsing and validation."""

    def test_from_dict_list(self):
        raw_list = [
            {"class": "person", "confidence": 0.88, "bbox": [0.2, 0.1, 0.5, 0.7]},
            {"class": "obstacle", "confidence": 0.72, "bbox": [0.6, 0.4, 0.8, 0.9]},
            {"class": "invalid", "confidence": 0.5, "bbox": [0.8, 0.2, 0.3, 0.5]},  # Reversed x
        ]
        detections = YOLOPerceptionAdapter.from_dict_list(raw_list)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].class_name, "person")
        self.assertAlmostEqual(detections[0].confidence, 0.88)
        self.assertEqual(detections[1].class_name, "obstacle")

    def test_from_mock_ultralytics_results(self):
        class MockBoxes:
            def __init__(self):
                self.xyxyn = [[0.1, 0.2, 0.4, 0.6], [0.5, 0.3, 0.8, 0.7]]
                self.conf = [0.85, 0.65]
                self.cls = [0, 1]

            def __len__(self):
                return len(self.cls)

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes()
                self.names = {0: "person", 1: "obstacle"}

        mock_res = MockResult()
        detections = YOLOPerceptionAdapter.from_ultralytics_results(mock_res)
        self.assertEqual(len(detections), 2)
        self.assertEqual(detections[0].class_name, "person")
        self.assertEqual(detections[1].class_name, "obstacle")


class TestGaitControllerIntegrationBridge(unittest.TestCase):
    """Tests for safe dispatch from high-level decisions to GaitController."""

    def setUp(self):
        self.mock_gait = MagicMock(spec=GaitController)
        self.bridge = NavigationGaitBridge(self.mock_gait, allow_backward=False)

    def test_forward_dispatch(self):
        decision = NavigationDecision(command=NavCommand.FORWARD, reason="Clear")
        cmd = self.bridge.dispatch(decision)
        self.assertEqual(cmd, "forward")
        self.mock_gait.set_command.assert_called_with("forward")

    def test_stop_dispatch(self):
        decision = NavigationDecision(command=NavCommand.STOP, reason="Obstacle")
        cmd = self.bridge.dispatch(decision)
        self.assertEqual(cmd, "stop")
        self.mock_gait.set_command.assert_called_with("stop")

    def test_wait_dispatch_safely_stops(self):
        decision = NavigationDecision(command=NavCommand.WAIT, reason="Sensor missing")
        cmd = self.bridge.dispatch(decision)
        self.assertEqual(cmd, "stop")
        self.mock_gait.set_command.assert_called_with("stop")

    def test_avoid_obstacle_left_dispatch(self):
        decision = NavigationDecision(
            command=NavCommand.AVOID_OBSTACLE,
            sub_action="TURN_LEFT",
            reason="Avoid"
        )
        cmd = self.bridge.dispatch(decision)
        self.assertEqual(cmd, "left")
        self.mock_gait.set_command.assert_called_with("left")

    def test_avoid_obstacle_right_dispatch(self):
        decision = NavigationDecision(
            command=NavCommand.AVOID_OBSTACLE,
            sub_action="TURN_RIGHT",
            reason="Avoid"
        )
        cmd = self.bridge.dispatch(decision)
        self.assertEqual(cmd, "right")
        self.mock_gait.set_command.assert_called_with("right")

    def test_unverified_backward_safely_blocked(self):
        # Unverified backward is disabled by default for hardware safety
        decision = NavigationDecision(command=NavCommand.BACKWARD, reason="Reverse")
        cmd = self.bridge.dispatch(decision)
        self.assertEqual(cmd, "stop")
        self.mock_gait.set_command.assert_called_with("stop")


if __name__ == "__main__":
    unittest.main()
