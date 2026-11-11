"""
SpotMicro Autonomous Navigation Decision Engine (Stage 1)
==========================================================
High-level navigation decision system for the SpotMicro quadruped robot.

SAFETY CONSTRAINTS & ARCHITECTURE:
- High-level discrete commands only: FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT,
  STOP, AVOID_OBSTACLE, WAIT.
- No direct control of servo PWM duty cycles or joint angles.
- Decoupled from low-level gait generation (app/gait_controller.py) and
  PCA9685 servo drivers (hardware/servo_controller.py).
- Hardware operations are completely disabled during software testing.
- Missing or invalid sensor readings are NEVER assumed to imply a clear path.
- Unverified reverse movement and automatic stair climbing are strictly disabled.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# High-Level Navigation Commands
# ---------------------------------------------------------------------------

class NavCommand(str, Enum):
    """Discrete high-level navigation commands emitted by the decision engine."""
    FORWARD = "FORWARD"
    BACKWARD = "BACKWARD"
    TURN_LEFT = "TURN_LEFT"
    TURN_RIGHT = "TURN_RIGHT"
    STOP = "STOP"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    WAIT = "WAIT"


# ---------------------------------------------------------------------------
# Bounding Box & Perception Structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundingBox:
    """Normalized bounding box with coordinates bounded in [0.0, 1.0]."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self):
        # Validate data types and finite numbers
        for name, val in [("x_min", self.x_min), ("y_min", self.y_min),
                          ("x_max", self.x_max), ("y_max", self.y_max)]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"BoundingBox {name} must be a number, got {type(val)}")
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"BoundingBox {name} must be finite, got {val}")

        # Validate coordinate order and range
        if not (0.0 <= self.x_min <= 1.0 and 0.0 <= self.x_max <= 1.0):
            raise ValueError(f"x coordinates must be in [0.0, 1.0]: [{self.x_min}, {self.x_max}]")
        if not (0.0 <= self.y_min <= 1.0 and 0.0 <= self.y_max <= 1.0):
            raise ValueError(f"y coordinates must be in [0.0, 1.0]: [{self.y_min}, {self.y_max}]")
        if self.x_min >= self.x_max:
            raise ValueError(f"x_min ({self.x_min}) must be strictly less than x_max ({self.x_max})")
        if self.y_min >= self.y_max:
            raise ValueError(f"y_min ({self.y_min}) must be strictly less than y_max ({self.y_max})")

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) / 2.0

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersects_horizontal_span(self, span_min: float, span_max: float) -> bool:
        """Checks if the bounding box horizontally overlaps the specified span."""
        return max(self.x_min, span_min) < min(self.x_max, span_max)

    def horizontal_region(self, left_bound: float = 0.33, right_bound: float = 0.67) -> str:
        """
        Determines rough bearing region in camera FOV.
        Returns 'LEFT', 'CENTER', or 'RIGHT'.
        Note: Camera regions are only a rough indication of direction,
        NOT a direct distance measurement.
        """
        cx = self.center_x
        if cx < left_bound and self.x_max <= 0.50:
            return "LEFT"
        elif cx > right_bound and self.x_min >= 0.50:
            return "RIGHT"
        return "CENTER"


@dataclass(frozen=True)
class Detection:
    """Structured perception detection output."""
    class_name: str
    confidence: float
    bbox: BoundingBox

    def __post_init__(self):
        if not isinstance(self.class_name, str) or not self.class_name.strip():
            raise ValueError(f"Detection class_name must be a non-empty string, got {self.class_name}")
        if not isinstance(self.confidence, (int, float)):
            raise TypeError(f"Detection confidence must be a number, got {type(self.confidence)}")
        if math.isnan(self.confidence) or math.isinf(self.confidence):
            raise ValueError(f"Detection confidence must be finite, got {self.confidence}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Detection confidence must be in [0.0, 1.0], got {self.confidence}")
        if not isinstance(self.bbox, BoundingBox):
            raise TypeError(f"Detection bbox must be an instance of BoundingBox, got {type(self.bbox)}")


# ---------------------------------------------------------------------------
# Sensor Reading Structures (Ultrasonic & IMU)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UltrasonicReading:
    """Distance measurement from an ultrasonic sensor (HC-SR04)."""
    distance_cm: Optional[float]
    timestamp: float = field(default_factory=time.time)
    is_valid: bool = True

    def __post_init__(self):
        if self.distance_cm is not None:
            if not isinstance(self.distance_cm, (int, float)):
                object.__setattr__(self, "is_valid", False)
                return
            if math.isnan(self.distance_cm) or math.isinf(self.distance_cm):
                object.__setattr__(self, "is_valid", False)
                return
            # HC-SR04 physical sensing limits: typically 2.0 cm to 400.0 cm
            if self.distance_cm < 2.0 or self.distance_cm > 400.0:
                object.__setattr__(self, "is_valid", False)


@dataclass
class UltrasonicSensors:
    """Collection of 4 HC-SR04 sensors: front, left, right, rear."""
    front: Optional[UltrasonicReading] = None
    left: Optional[UltrasonicReading] = None
    right: Optional[UltrasonicReading] = None
    rear: Optional[UltrasonicReading] = None

    def get_front_distance(self, max_age_s: float, current_time: float) -> Tuple[Optional[float], bool, str]:
        """
        Returns (distance_cm, is_valid_and_fresh, reason).
        Missing or stale data is flagged explicitly.
        """
        if self.front is None:
            return None, False, "Front ultrasonic reading is missing"
        if not self.front.is_valid or self.front.distance_cm is None:
            return None, False, "Front ultrasonic reading is invalid or out-of-range"
        age = current_time - self.front.timestamp
        if age < 0 or age > max_age_s:
            return self.front.distance_cm, False, f"Front ultrasonic data is stale ({age:.3f}s old)"
        return self.front.distance_cm, True, "OK"

    def get_lateral_clearance(
        self,
        side: str,
        clearance_threshold_cm: float,
        max_age_s: float,
        current_time: float
    ) -> Tuple[bool, Optional[float], str]:
        """
        Checks if a lateral side ('left' or 'right') is clear.
        Returns (is_clear, distance_cm, reason).
        Rule: If sensor reading is missing or invalid, clearance CANNOT be assumed.
        """
        sensor = self.left if side == "left" else self.right
        if sensor is None:
            return False, None, f"{side.capitalize()} ultrasonic sensor is missing"
        if not sensor.is_valid or sensor.distance_cm is None:
            return False, None, f"{side.capitalize()} ultrasonic reading is invalid"
        age = current_time - sensor.timestamp
        if age < 0 or age > max_age_s:
            return False, sensor.distance_cm, f"{side.capitalize()} ultrasonic reading is stale ({age:.3f}s old)"
        if sensor.distance_cm < clearance_threshold_cm:
            return False, sensor.distance_cm, f"{side.capitalize()} side obstructed ({sensor.distance_cm:.1f} cm < {clearance_threshold_cm:.1f} cm)"
        return True, sensor.distance_cm, f"{side.capitalize()} side clear ({sensor.distance_cm:.1f} cm)"


@dataclass(frozen=True)
class IMUReading:
    """Orientation and stability status from BNO055 IMU."""
    roll: Optional[float] = None       # Tilt left/right in degrees
    pitch: Optional[float] = None      # Tilt nose up/down in degrees
    yaw: Optional[float] = None        # Heading in degrees
    is_stable: bool = True             # Dynamic stability flag
    timestamp: float = field(default_factory=time.time)

    def is_safe(self, max_roll_deg: float, max_pitch_deg: float, max_age_s: float, current_time: float) -> Tuple[bool, str]:
        """Validates IMU safety limits (excessive tilt or instability)."""
        age = current_time - self.timestamp
        if age < 0 or age > max_age_s:
            return False, f"IMU data is stale ({age:.3f}s old)"
        if not self.is_stable:
            return False, "IMU reports dynamic instability / tipping hazard"
        if self.roll is not None:
            if math.isnan(self.roll) or math.isinf(self.roll):
                return False, "IMU roll reading is non-finite"
            if abs(self.roll) > max_roll_deg:
                return False, f"Unsafe lateral roll tilt: {self.roll:.1f}° (max {max_roll_deg:.1f}°)"
        if self.pitch is not None:
            if math.isnan(self.pitch) or math.isinf(self.pitch):
                return False, "IMU pitch reading is non-finite"
            if abs(self.pitch) > max_pitch_deg:
                return False, f"Unsafe pitch tilt: {self.pitch:.1f}° (max {max_pitch_deg:.1f}°)"
        return True, "IMU orientation stable"


# ---------------------------------------------------------------------------
# Navigation Input and Decision Structs
# ---------------------------------------------------------------------------

@dataclass
class NavigationInput:
    """Structured perception and telemetry input for one decision cycle."""
    detections: List[Detection] = field(default_factory=list)
    ultrasonic: UltrasonicSensors = field(default_factory=UltrasonicSensors)
    imu: Optional[IMUReading] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class NavigationDecision:
    """Output decision emitted by the Navigation Decision Engine."""
    command: NavCommand
    reason: str
    timestamp: float = field(default_factory=time.time)
    sub_action: Optional[str] = None  # e.g. "TURN_LEFT" or "TURN_RIGHT" during AVOID_OBSTACLE
    telemetry: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Configurable Navigation Parameters
# ---------------------------------------------------------------------------

@dataclass
class NavigationConfig:
    """Configuration parameters and safety thresholds for navigation."""

    # YOLO Confidence Thresholds per class
    # Model classes: 0: person, 1: obstacle, 2: chair, 3: stairs, 4: door, 5: robot
    # Note known limitations: chair detection is weak (lower threshold),
    # stairs have insufficient examples (lower threshold for safety),
    # door is not assumed traversable.
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "person": 0.40,
        "obstacle": 0.35,
        "chair": 0.30,
        "stairs": 0.30,
        "door": 0.40,
        "robot": 0.40,
        "default": 0.35,
    })

    # Ultrasonic Distance Thresholds (cm)
    front_stop_distance_cm: float = 30.0       # Emergency STOP if front object is closer than this
    front_caution_distance_cm: float = 65.0    # Begin obstacle avoidance/steering if closer than this
    side_clearance_distance_cm: float = 35.0   # Required clearance on side to allow avoidance turn
    min_valid_distance_cm: float = 2.0
    max_valid_distance_cm: float = 400.0

    # Camera Central Corridor Definition (fraction of image width [0.0, 1.0])
    corridor_x_min: float = 0.25               # Objects intersecting [0.25, 0.75] are in the center path
    corridor_x_max: float = 0.75
    critical_person_corridor_min: float = 0.20
    critical_person_corridor_max: float = 0.80

    # IMU Safety Limits
    max_roll_deg: float = 20.0                 # Degrees tilt before emergency STOP
    max_pitch_deg: float = 20.0

    # Timing, Staleness and Dead-man Policy
    max_sensor_age_s: float = 0.5              # Readings older than 500 ms are considered stale
    deadman_timeout_s: float = 1.0             # If no fresh update within 1s, trigger STOP
    require_front_ultrasonic: bool = True      # Rule 9: Never assume path is clear without sensors

    # Hysteresis and Oscillation Damping
    consecutive_frames_required: int = 2       # Frames required to switch active movement command
    stop_hold_frames: int = 2                  # Hold STOP for at least N cycles before resuming

    # Safe Motion Enforcements
    allow_unverified_reverse: bool = False     # Strictly false: reverse motion unverified on hardware
    allow_automatic_stairs: bool = False       # Strictly false: quadruped stair climbing unverified

    def get_confidence_threshold(self, class_name: str) -> float:
        return self.confidence_thresholds.get(class_name.lower(), self.confidence_thresholds.get("default", 0.35))


# ---------------------------------------------------------------------------
# Navigation Decision Engine
# ---------------------------------------------------------------------------

class NavigationDecisionEngine:
    """
    Modular Navigation Decision Engine for SpotMicro.
    Evaluates multimodal sensor inputs and issues deterministic high-level commands.
    """

    def __init__(self, config: Optional[NavigationConfig] = None):
        self.config = config or NavigationConfig()
        self._last_cycle_time: Optional[float] = None
        self._current_command: NavCommand = NavCommand.STOP
        self._candidate_command: NavCommand = NavCommand.STOP
        self._candidate_count: int = 0
        self._stop_hold_counter: int = 0
        self._last_decision_reason: str = "Initialized in safe STOP state"

    @property
    def current_command(self) -> NavCommand:
        return self._current_command

    def reset(self):
        """Resets internal state machine to safe initial STOP."""
        self._last_cycle_time = None
        self._current_command = NavCommand.STOP
        self._candidate_command = NavCommand.STOP
        self._candidate_count = 0
        self._stop_hold_counter = 0
        self._last_decision_reason = "Reset to safe STOP state"

    def decide(self, nav_input: NavigationInput) -> NavigationDecision:
        """
        Executes a single navigation decision cycle.
        Returns a validated NavigationDecision.
        """
        current_time = nav_input.timestamp if nav_input.timestamp is not None else time.time()
        telemetry: Dict[str, Any] = {
            "timestamp": current_time,
            "raw_candidate": None,
            "filter_applied": False,
        }

        # -------------------------------------------------------------------
        # 1. Dead-Man Policy / Timeout Check
        # -------------------------------------------------------------------
        if self._last_cycle_time is not None:
            dt = current_time - self._last_cycle_time
            if dt > self.config.deadman_timeout_s:
                self._current_command = NavCommand.STOP
                self._candidate_command = NavCommand.STOP
                self._candidate_count = 0
                self._stop_hold_counter = self.config.stop_hold_frames
                reason = f"Dead-man timeout triggered: pipeline stalled for {dt:.3f}s (> {self.config.deadman_timeout_s}s)"
                self._last_decision_reason = reason
                self._last_cycle_time = current_time
                telemetry["deadman_triggered"] = True
                return NavigationDecision(command=NavCommand.STOP, reason=reason, timestamp=current_time, telemetry=telemetry)

        self._last_cycle_time = current_time

        # -------------------------------------------------------------------
        # 2. IMU Stability & Tilt Safety Check (Highest Priority)
        # -------------------------------------------------------------------
        if nav_input.imu is not None:
            imu_safe, imu_reason = nav_input.imu.is_safe(
                max_roll_deg=self.config.max_roll_deg,
                max_pitch_deg=self.config.max_pitch_deg,
                max_age_s=self.config.max_sensor_age_s,
                current_time=current_time
            )
            telemetry["imu_status"] = {"safe": imu_safe, "reason": imu_reason}
            if not imu_safe:
                return self._immediate_emergency_stop(f"IMU Safety Alert: {imu_reason}", current_time, telemetry)

        # -------------------------------------------------------------------
        # 3. Filter Valid Perception Detections by Confidence
        # -------------------------------------------------------------------
        valid_detections: List[Detection] = []
        for det in nav_input.detections:
            if not isinstance(det, Detection):
                continue
            thresh = self.config.get_confidence_threshold(det.class_name)
            if det.confidence >= thresh:
                valid_detections.append(det)

        telemetry["valid_detections_count"] = len(valid_detections)
        telemetry["detections"] = [
            {"class": d.class_name, "conf": round(d.confidence, 3), "region": d.bbox.horizontal_region()}
            for d in valid_detections
        ]

        # -------------------------------------------------------------------
        # 4. Check for Detected People in Intended Path (Immediate STOP)
        # -------------------------------------------------------------------
        for det in valid_detections:
            if det.class_name.lower() == "person":
                if det.bbox.intersects_horizontal_span(
                    self.config.critical_person_corridor_min,
                    self.config.critical_person_corridor_max
                ):
                    return self._immediate_emergency_stop(
                        f"Person detected in intended center corridor (confidence: {det.confidence:.2f})",
                        current_time,
                        telemetry
                    )

        # -------------------------------------------------------------------
        # 5. Check for Stairs in Intended Path (Automatic Climbing Forbidden)
        # -------------------------------------------------------------------
        for det in valid_detections:
            if det.class_name.lower() == "stairs":
                if det.bbox.intersects_horizontal_span(self.config.corridor_x_min, self.config.corridor_x_max):
                    if not self.config.allow_automatic_stairs:
                        return self._immediate_emergency_stop(
                            f"Stairs detected ahead (confidence: {det.confidence:.2f}); automatic stair climbing is disabled for safety",
                            current_time,
                            telemetry
                        )

        # -------------------------------------------------------------------
        # 6. Front Ultrasonic Distance & Sensor Reliability Check
        # -------------------------------------------------------------------
        front_dist, front_valid, front_reason = nav_input.ultrasonic.get_front_distance(
            max_age_s=self.config.max_sensor_age_s,
            current_time=current_time
        )
        telemetry["front_ultrasonic"] = {"distance_cm": front_dist, "valid": front_valid, "reason": front_reason}

        # Rule 9: Missing or invalid sensor readings must never be interpreted
        # as proof that the path is clear.
        if self.config.require_front_ultrasonic and not front_valid:
            return self._apply_hysteresis(
                NavCommand.WAIT,
                f"Cannot verify path clearance: {front_reason}",
                current_time,
                telemetry
            )

        # Immediate Proximity Emergency Stop via Ultrasonic
        if front_dist is not None and front_dist <= self.config.front_stop_distance_cm:
            return self._immediate_emergency_stop(
                f"Front obstacle critically close ({front_dist:.1f} cm <= {self.config.front_stop_distance_cm:.1f} cm)",
                current_time,
                telemetry
            )

        # -------------------------------------------------------------------
        # 7. Evaluate Center Path Obstruction (Perception + Ultrasonic)
        # -------------------------------------------------------------------
        center_path_obstructed = False
        obstruction_reasons: List[str] = []

        # A. Ultrasonic Caution Range
        if front_dist is not None and front_dist < self.config.front_caution_distance_cm:
            center_path_obstructed = True
            obstruction_reasons.append(f"Front ultrasonic caution proximity ({front_dist:.1f} cm < {self.config.front_caution_distance_cm:.1f} cm)")

        # B. YOLO Detections in Center Corridor (obstacle, chair, robot)
        for det in valid_detections:
            cls = det.class_name.lower()
            if cls in ("obstacle", "chair", "robot"):
                if det.bbox.intersects_horizontal_span(self.config.corridor_x_min, self.config.corridor_x_max):
                    center_path_obstructed = True
                    obstruction_reasons.append(f"{cls} detected in center path (conf: {det.confidence:.2f})")

        # C. Door Handling: A detected door is not proof that the doorway is traversable.
        for det in valid_detections:
            if det.class_name.lower() == "door":
                if det.bbox.intersects_horizontal_span(self.config.corridor_x_min, self.config.corridor_x_max):
                    # If ultrasonic reading doesn't confirm large open depth ahead, treat door with caution
                    if front_dist is None or front_dist < self.config.front_caution_distance_cm:
                        center_path_obstructed = True
                        obstruction_reasons.append("Door detected ahead but clearance depth is not verified")

        # -------------------------------------------------------------------
        # 8. Obstacle Avoidance or Turn Decision
        # -------------------------------------------------------------------
        if center_path_obstructed:
            left_clear, left_dist, left_reason = nav_input.ultrasonic.get_lateral_clearance(
                "left",
                self.config.side_clearance_distance_cm,
                self.config.max_sensor_age_s,
                current_time
            )
            right_clear, right_dist, right_reason = nav_input.ultrasonic.get_lateral_clearance(
                "right",
                self.config.side_clearance_distance_cm,
                self.config.max_sensor_age_s,
                current_time
            )

            # Also check if YOLO detects lateral obstacles
            left_camera_blocked = any(
                d.bbox.horizontal_region() == "LEFT" and d.class_name.lower() in ("obstacle", "person", "chair", "robot")
                for d in valid_detections
            )
            right_camera_blocked = any(
                d.bbox.horizontal_region() == "RIGHT" and d.class_name.lower() in ("obstacle", "person", "chair", "robot")
                for d in valid_detections
            )

            effective_left_safe = left_clear and not left_camera_blocked
            effective_right_safe = right_clear and not right_camera_blocked

            telemetry["avoidance_eval"] = {
                "effective_left_safe": effective_left_safe,
                "left_clear": left_clear,
                "left_dist": left_dist,
                "left_camera_blocked": left_camera_blocked,
                "effective_right_safe": effective_right_safe,
                "right_clear": right_clear,
                "right_dist": right_dist,
                "right_camera_blocked": right_camera_blocked,
            }

            # If both sides are blocked: Cannot safely maneuver
            if not effective_left_safe and not effective_right_safe:
                return self._apply_hysteresis(
                    NavCommand.STOP,
                    f"Path blocked ({'; '.join(obstruction_reasons)}) and both lateral corridors unavailable (Left: {left_reason}; Right: {right_reason})",
                    current_time,
                    telemetry
                )

            # Prefer turning toward the side with greater ultrasonic clearance
            if effective_left_safe and effective_right_safe:
                ld = left_dist if left_dist is not None else 0.0
                rd = right_dist if right_dist is not None else 0.0
                if ld >= rd:
                    chosen_turn = NavCommand.TURN_LEFT
                    sub_act = "TURN_LEFT"
                    turn_reason = f"Obstacle in center ({'; '.join(obstruction_reasons)}); steering LEFT (clearance: {ld:.1f} cm vs right {rd:.1f} cm)"
                else:
                    chosen_turn = NavCommand.TURN_RIGHT
                    sub_act = "TURN_RIGHT"
                    turn_reason = f"Obstacle in center ({'; '.join(obstruction_reasons)}); steering RIGHT (clearance: {rd:.1f} cm vs left {ld:.1f} cm)"
            elif effective_left_safe:
                chosen_turn = NavCommand.TURN_LEFT
                sub_act = "TURN_LEFT"
                turn_reason = f"Obstacle in center ({'; '.join(obstruction_reasons)}); right side obstructed ({right_reason}); steering LEFT ({left_dist} cm)"
            else:
                chosen_turn = NavCommand.TURN_RIGHT
                sub_act = "TURN_RIGHT"
                turn_reason = f"Obstacle in center ({'; '.join(obstruction_reasons)}); left side obstructed ({left_reason}); steering RIGHT ({right_dist} cm)"

            telemetry["sub_action"] = sub_act
            return self._apply_hysteresis(
                NavCommand.AVOID_OBSTACLE,
                turn_reason,
                current_time,
                telemetry,
                sub_action=sub_act
            )

        # -------------------------------------------------------------------
        # 9. Path is Clear: Advance FORWARD
        # -------------------------------------------------------------------
        forward_reason = "Path verified clear: front ultrasonic and camera corridor show no hazards"
        if front_dist is not None:
            forward_reason += f" (front clearance: {front_dist:.1f} cm)"

        return self._apply_hysteresis(
            NavCommand.FORWARD,
            forward_reason,
            current_time,
            telemetry
        )

    # -----------------------------------------------------------------------
    # Internal Hysteresis & Emergency Handling Methods
    # -----------------------------------------------------------------------

    def _immediate_emergency_stop(self, reason: str, timestamp: float, telemetry: Dict[str, Any]) -> NavigationDecision:
        """
        Immediately activates STOP with 0 delay (bypassing hysteresis).
        Safety-critical stops must never wait for consecutive frames!
        """
        self._current_command = NavCommand.STOP
        self._candidate_command = NavCommand.STOP
        self._candidate_count = 0
        self._stop_hold_counter = self.config.stop_hold_frames
        self._last_decision_reason = reason
        telemetry["emergency_stop"] = True
        return NavigationDecision(command=NavCommand.STOP, reason=reason, timestamp=timestamp, telemetry=telemetry)

    def _apply_hysteresis(
        self,
        candidate: NavCommand,
        reason: str,
        timestamp: float,
        telemetry: Dict[str, Any],
        sub_action: Optional[str] = None
    ) -> NavigationDecision:
        """
        Applies consecutive-frame confirmation and hold-time hysteresis
        to damp rapid command switching while guaranteeing swift safety responses.
        """
        telemetry["raw_candidate"] = candidate.value

        # Enforce STOP hold frames: prevent moving until robot has settled
        if self._current_command == NavCommand.STOP and candidate != NavCommand.STOP:
            if self._stop_hold_counter > 0:
                self._stop_hold_counter -= 1
                hold_reason = f"Holding STOP for post-stop stability ({self._stop_hold_counter + 1} frames remaining); candidate was {candidate.value}"
                telemetry["filter_applied"] = True
                telemetry["stop_held"] = True
                return NavigationDecision(
                    command=NavCommand.STOP,
                    reason=hold_reason,
                    timestamp=timestamp,
                    telemetry=telemetry
                )

        # Immediate reaction if candidate is STOP
        if candidate == NavCommand.STOP:
            self._current_command = NavCommand.STOP
            self._candidate_command = NavCommand.STOP
            self._candidate_count = 0
            self._stop_hold_counter = self.config.stop_hold_frames
            self._last_decision_reason = reason
            return NavigationDecision(command=NavCommand.STOP, reason=reason, timestamp=timestamp, telemetry=telemetry)

        # If candidate matches current active command, maintain it
        if candidate == self._current_command:
            self._candidate_command = candidate
            self._candidate_count = 0
            self._last_decision_reason = reason
            return NavigationDecision(
                command=self._current_command,
                reason=reason,
                timestamp=timestamp,
                sub_action=sub_action,
                telemetry=telemetry
            )

        # New candidate: track consecutive occurrences
        if candidate == self._candidate_command:
            self._candidate_count += 1
        else:
            self._candidate_command = candidate
            self._candidate_count = 1

        # Check if candidate met confirmation threshold
        if self._candidate_count >= self.config.consecutive_frames_required:
            self._current_command = candidate
            self._last_decision_reason = reason
            telemetry["confirmed_transition"] = True
            return NavigationDecision(
                command=self._current_command,
                reason=reason,
                timestamp=timestamp,
                sub_action=sub_action,
                telemetry=telemetry
            )
        else:
            # Not confirmed yet: hold previous command
            filtered_reason = (
                f"Damped transition to {candidate.value} ({self._candidate_count}/{self.config.consecutive_frames_required} frames); "
                f"holding {self._current_command.value}."
            )
            telemetry["filter_applied"] = True
            return NavigationDecision(
                command=self._current_command,
                reason=filtered_reason,
                timestamp=timestamp,
                sub_action=sub_action,
                telemetry=telemetry
            )


# ---------------------------------------------------------------------------
# YOLO Perception Adapter
# ---------------------------------------------------------------------------

class YOLOPerceptionAdapter:
    """
    Perception adapter converting Ultralytics YOLO results or raw detection
    dictionaries into validated Detection objects for the navigation engine.
    """

    CLASS_MAPPING: Dict[int, str] = {
        0: "person",
        1: "obstacle",
        2: "chair",
        3: "stairs",
        4: "door",
        5: "robot",
    }

    @classmethod
    def from_ultralytics_results(
        cls,
        results: Any,
        image_shape: Optional[Tuple[int, int]] = None
    ) -> List[Detection]:
        """
        Converts Ultralytics YOLO Results object into structured List[Detection].
        Handles normalized coordinates (xyxyn) or pixel coordinates (xyxy).
        """
        detections: List[Detection] = []
        if results is None:
            return detections

        # Support single Results object or iterable of Results
        results_list = results if isinstance(results, (list, tuple)) else [results]

        for res in results_list:
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue

            try:
                if len(boxes) == 0:
                    continue
            except TypeError:
                pass

            # Class mapping from Results if available
            names = getattr(res, "names", cls.CLASS_MAPPING)

            # Extract normalized xyxy or pixel xyxy
            has_xyxyn = hasattr(boxes, "xyxyn") and boxes.xyxyn is not None
            xy_data = boxes.xyxyn if has_xyxyn else boxes.xyxy
            conf_data = boxes.conf
            cls_data = boxes.cls

            # Convert tensors/arrays to standard Python lists
            if hasattr(xy_data, "cpu"):
                xy_data = xy_data.cpu().numpy()
            if hasattr(conf_data, "cpu"):
                conf_data = conf_data.cpu().numpy()
            if hasattr(cls_data, "cpu"):
                cls_data = cls_data.cpu().numpy()

            img_h, img_w = (1.0, 1.0)
            if not has_xyxyn:
                if image_shape is not None:
                    img_h, img_w = float(image_shape[0]), float(image_shape[1])
                elif hasattr(res, "orig_shape") and res.orig_shape is not None:
                    img_h, img_w = float(res.orig_shape[0]), float(res.orig_shape[1])

            for i in range(len(xy_data)):
                try:
                    coords = xy_data[i]
                    conf = float(conf_data[i])
                    class_id = int(cls_data[i])
                    class_name = names.get(class_id, cls.CLASS_MAPPING.get(class_id, f"class_{class_id}"))

                    if has_xyxyn:
                        x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                    else:
                        x1 = float(coords[0]) / img_w
                        y1 = float(coords[1]) / img_h
                        x2 = float(coords[2]) / img_w
                        y2 = float(coords[3]) / img_h

                    # Clamp minor precision artifacts
                    x1 = max(0.0, min(1.0, x1))
                    y1 = max(0.0, min(1.0, y1))
                    x2 = max(0.0, min(1.0, x2))
                    y2 = max(0.0, min(1.0, y2))

                    bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
                    detections.append(Detection(class_name=class_name, confidence=conf, bbox=bbox))
                except (ValueError, TypeError, ZeroDivisionError):
                    # Reject malformed boxes safely
                    continue

        return detections

    @classmethod
    def from_dict_list(cls, dict_list: List[Dict[str, Any]]) -> List[Detection]:
        """
        Converts a list of dicts (e.g. from JSON telemetry or test mocks) into Detections.
        Expected format:
        {'class': 'person', 'confidence': 0.88, 'bbox': [x1, y1, x2, y2]}
        """
        detections: List[Detection] = []
        if not dict_list:
            return detections

        for item in dict_list:
            try:
                class_name = item.get("class") or item.get("class_name")
                confidence = float(item.get("confidence", 0.0))
                bbox_data = item.get("bbox")

                if bbox_data is None or len(bbox_data) != 4:
                    continue

                x1, y1, x2, y2 = float(bbox_data[0]), float(bbox_data[1]), float(bbox_data[2]), float(bbox_data[3])
                bbox = BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2)
                detections.append(Detection(class_name=str(class_name), confidence=confidence, bbox=bbox))
            except (ValueError, TypeError):
                continue

        return detections


# ---------------------------------------------------------------------------
# Gait Controller Integration Bridge
# ---------------------------------------------------------------------------

class NavigationGaitBridge:
    """
    Integration Bridge between NavigationDecisionEngine and GaitController.
    Enforces architectural safety boundary:
    - Never commands individual servos or PWM duty cycles.
    - Maps discrete high-level commands to allowed GaitController strings:
      'forward', 'backward', 'left', 'right', 'stop'.
    - Disallows unverified reverse movement.
    """

    GAIT_ALLOWED_COMMANDS = {"forward", "backward", "left", "right", "stop"}

    def __init__(self, gait_controller: Any, allow_backward: bool = False):
        self.gait_controller = gait_controller
        self.allow_backward = allow_backward

    def dispatch(self, decision: NavigationDecision) -> str:
        """
        Translates NavigationDecision into a validated GaitController command
        and applies it. Returns the executed low-level gait command.
        """
        cmd = decision.command

        if cmd in (NavCommand.STOP, NavCommand.WAIT):
            gait_cmd = "stop"
        elif cmd == NavCommand.FORWARD:
            gait_cmd = "forward"
        elif cmd == NavCommand.TURN_LEFT:
            gait_cmd = "left"
        elif cmd == NavCommand.TURN_RIGHT:
            gait_cmd = "right"
        elif cmd == NavCommand.AVOID_OBSTACLE:
            # Map sub_action if provided; fallback to stop if evasive direction undetermined
            if decision.sub_action == "TURN_LEFT":
                gait_cmd = "left"
            elif decision.sub_action == "TURN_RIGHT":
                gait_cmd = "right"
            else:
                gait_cmd = "stop"
        elif cmd == NavCommand.BACKWARD:
            if not self.allow_backward:
                # Safe fallback: unverified reverse movement is forbidden
                gait_cmd = "stop"
            else:
                gait_cmd = "backward"
        else:
            gait_cmd = "stop"

        if gait_cmd not in self.GAIT_ALLOWED_COMMANDS:
            gait_cmd = "stop"

        # Apply to gait controller safely
        if self.gait_controller is not None and hasattr(self.gait_controller, "set_command"):
            self.gait_controller.set_command(gait_cmd)

        return gait_cmd
