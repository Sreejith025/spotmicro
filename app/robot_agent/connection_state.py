"""
SpotMicro Robot Agent - Connection State Handling
=================================================
Represents the discrete connection and operating states of the Raspberry Pi
robot software agent communicating with the SpotMicro web backend.

Strictly software-only: NO physical hardware or actuator access.
"""

from __future__ import annotations

from enum import Enum


class ConnectionState(str, Enum):
    """
    Discrete connection states for the robot agent.
    """
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    AUTHENTICATED = "AUTHENTICATED"
    SAFE_IDLE = "SAFE_IDLE"
    ERROR = "ERROR"

