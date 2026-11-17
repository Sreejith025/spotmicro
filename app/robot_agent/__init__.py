"""
SpotMicro Robot Agent Package (Stage 2.1)
=========================================
Raspberry Pi 5 secure software connection module.
Provides authenticated communication with the SpotMicro web backend,
connection state tracking, software watchdog, and local safety authority.

SAFETY CONSTRAINTS:
- Strictly SOFTWARE ONLY.
- hardware_active == False.
- No GPIO, PCA9685, or servo PWM operations.
"""

from app.robot_agent.config import (
    ConfigurationError,
    RobotAgentConfig,
    load_agent_config,
)
from app.robot_agent.connection_state import ConnectionState
from app.robot_agent.safety_authority import LocalSafetyAuthority, NavCommand
from app.robot_agent.watchdog import SoftwareWatchdog
from app.robot_agent.client import RobotAgent

__all__ = [
    "ConfigurationError",
    "RobotAgentConfig",
    "load_agent_config",
    "ConnectionState",
    "LocalSafetyAuthority",
    "NavCommand",
    "SoftwareWatchdog",
    "RobotAgent",
]

