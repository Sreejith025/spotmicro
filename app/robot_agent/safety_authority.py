"""
SpotMicro Robot Agent - Local Safety Authority
==============================================
Enforces high-level safety constraints on the Raspberry Pi robot agent.
Decoupled completely from physical hardware, PCA9685, and servo PWM.

SAFETY CONSTRAINTS:
- High-level discrete commands ONLY:
    FORWARD, BACKWARD, TURN_LEFT, TURN_RIGHT, STOP, AVOID_OBSTACLE, WAIT
- Strictly NO servo angles, duty cycles, or joint control.
- Hardware is strictly INACTIVE (hardware_active == False).
- Malformed, unexpected, or unauthenticated commands immediately trigger
  SAFE_IDLE / STOP.
- Emergency Stop overrides all movement.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional, Set, Tuple

try:
    from app.navigation import NavCommand
except ImportError:
    from navigation import NavCommand

logger = logging.getLogger("SpotMicroRobotAgent.SafetyAuthority")

# Supported command strings mapped to canonical NavCommand enum
COMMAND_MAP: Dict[str, NavCommand] = {
    # Exact NavCommand values
    "FORWARD": NavCommand.FORWARD,
    "BACKWARD": NavCommand.BACKWARD,
    "TURN_LEFT": NavCommand.TURN_LEFT,
    "TURN_RIGHT": NavCommand.TURN_RIGHT,
    "STOP": NavCommand.STOP,
    "AVOID_OBSTACLE": NavCommand.AVOID_OBSTACLE,
    "WAIT": NavCommand.WAIT,
    # Web dashboard lowercase equivalents
    "forward": NavCommand.FORWARD,
    "backward": NavCommand.BACKWARD,
    "left": NavCommand.TURN_LEFT,
    "right": NavCommand.TURN_RIGHT,
    "turn_left": NavCommand.TURN_LEFT,
    "turn_right": NavCommand.TURN_RIGHT,
    "stop": NavCommand.STOP,
    "stand": NavCommand.WAIT,
    "sit": NavCommand.STOP,
    "wait": NavCommand.WAIT,
    "avoid_obstacle": NavCommand.AVOID_OBSTACLE,
}


class LocalSafetyAuthority:
    """
    Arbitrates and enforces safety on the robot agent before commands can reach
    any motion generator. Decoupled from physical hardware.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._current_command: NavCommand = NavCommand.STOP
        self._is_safe_idle: bool = True
        self._emergency_stop: bool = False
        self._last_command_time: float = time.time()
        self._hardware_active: bool = False  # Strictly False in Stage 2.1

    @property
    def hardware_active(self) -> bool:
        """Hardware is strictly disabled during software connection stages."""
        return self._hardware_active

    @property
    def current_command(self) -> NavCommand:
        with self._lock:
            return self._current_command

    @property
    def is_safe_idle(self) -> bool:
        with self._lock:
            return self._is_safe_idle

    @property
    def emergency_stop(self) -> bool:
        with self._lock:
            return self._emergency_stop

    @property
    def last_command_time(self) -> float:
        with self._lock:
            return self._last_command_time

    def enter_safe_stop(self, reason: str = "Safe stop requested") -> NavCommand:
        """
        Forces the robot into SAFE_IDLE and commands STOP.
        Called on disconnections, watchdog timeouts, or communication errors.
        """
        with self._lock:
            self._current_command = NavCommand.STOP
            self._is_safe_idle = True
            self._last_command_time = time.time()
            logger.info(f"LocalSafetyAuthority: SAFE_IDLE / STOP activated ({reason})")
            return self._current_command

    def trigger_emergency_stop(self, reason: str = "Emergency stop triggered") -> NavCommand:
        """
        Immediately locks the safety authority in Emergency Stop state.
        Overrides any movement and commands STOP.
        """
        with self._lock:
            self._emergency_stop = True
            self._current_command = NavCommand.STOP
            self._is_safe_idle = True
            self._last_command_time = time.time()
            logger.warning(f"LocalSafetyAuthority: EMERGENCY STOP ({reason})")
            return self._current_command

    def reset_emergency_stop(self) -> bool:
        """
        Clears the emergency stop lock. Remains in SAFE_IDLE / STOP until
        a new valid command is processed.
        """
        with self._lock:
            self._emergency_stop = False
            self._current_command = NavCommand.STOP
            self._is_safe_idle = True
            logger.info("LocalSafetyAuthority: Emergency stop reset. In SAFE_IDLE / STOP.")
            return True

    def process_command(self, payload: Any) -> Tuple[bool, NavCommand, str]:
        """
        Validates an incoming command payload from the backend.
        Rejects malformed, non-dict, non-string, or unsupported commands,
        immediately falling back to SAFE_IDLE and STOP.

        Returns: (success: bool, command: NavCommand, message: str)
        """
        with self._lock:
            if self._emergency_stop:
                self._current_command = NavCommand.STOP
                self._is_safe_idle = True
                return False, NavCommand.STOP, "Emergency Stop is active. Movement blocked."

            raw_cmd: Optional[str] = None
            if isinstance(payload, dict):
                cmd_val = payload.get("command")
                if isinstance(cmd_val, str):
                    raw_cmd = cmd_val
                else:
                    logger.warning(f"Malformed command payload: 'command' field is not string: {cmd_val}")
            elif isinstance(payload, str):
                raw_cmd = payload
            else:
                logger.warning(f"Malformed command payload type: {type(payload)}")

            if not raw_cmd:
                self._current_command = NavCommand.STOP
                self._is_safe_idle = True
                return False, NavCommand.STOP, "Malformed command payload: missing valid command string."

            normalized = raw_cmd.strip()
            mapped_cmd = COMMAND_MAP.get(normalized) or COMMAND_MAP.get(normalized.upper())

            if mapped_cmd is None:
                self._current_command = NavCommand.STOP
                self._is_safe_idle = True
                logger.warning(f"Unknown or disallowed command rejected: '{raw_cmd}'. Fallback to SAFE / STOP.")
                return False, NavCommand.STOP, f"Unknown or disallowed command: '{raw_cmd}'"

            self._current_command = mapped_cmd
            self._is_safe_idle = (mapped_cmd in (NavCommand.STOP, NavCommand.WAIT))
            self._last_command_time = time.time()
            logger.info(f"LocalSafetyAuthority: Dispatched valid high-level command '{mapped_cmd.value}'")
            return True, mapped_cmd, f"Command '{mapped_cmd.value}' accepted."

    def get_status(self) -> Dict[str, Any]:
        """Returns snapshot of current safety authority state."""
        with self._lock:
            return {
                "current_command": self._current_command.value,
                "is_safe_idle": self._is_safe_idle,
                "emergency_stop": self._emergency_stop,
                "hardware_active": self._hardware_active,
                "last_command_time": self._last_command_time,
            }

