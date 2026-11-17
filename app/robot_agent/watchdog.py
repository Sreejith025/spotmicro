"""
SpotMicro Robot Agent - Software Watchdog Abstraction
====================================================
Pure software watchdog monitoring backend communication heartbeat and
command freshness.

SAFETY CONSTRAINTS:
- Strictly SOFTWARE ONLY.
- NO connection to hardware watchdogs, GPIO pins, relays, or motors.
- Immediately calls on_stale callback (transitioning robot to SAFE_IDLE / STOP)
  if the connection or heartbeat becomes stale.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger("SpotMicroRobotAgent.Watchdog")


class SoftwareWatchdog:
    """
    Thread-safe software-only watchdog timer.
    Monitors periodic kicks/heartbeats from the web backend or agent loop.
    Triggers on_stale_callback when timeout expires.
    """

    def __init__(
        self,
        timeout_s: float = 5.0,
        on_stale_callback: Optional[Callable[[], None]] = None,
        check_interval_s: float = 0.2,
    ):
        if timeout_s <= 0:
            raise ValueError("Watchdog timeout_s must be strictly positive.")
        
        self.timeout_s = float(timeout_s)
        self.check_interval_s = float(check_interval_s)
        self.on_stale_callback = on_stale_callback

        self._lock = threading.Lock()
        self._last_kick = time.time()
        self._is_stale = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    @property
    def is_stale(self) -> bool:
        with self._lock:
            return self._is_stale

    @property
    def last_kick_time(self) -> float:
        with self._lock:
            return self._last_kick

    def feed(self):
        """
        Kicks / feeds the watchdog timer, resetting the timeout counter.
        Can be called upon receiving WebSocket messages, heartbeat pongs, or valid commands.
        """
        with self._lock:
            self._last_kick = time.time()
            if self._is_stale:
                logger.info("Watchdog recovered from stale state via fresh heartbeat.")
                self._is_stale = False

    def check(self) -> bool:
        """
        Checks whether the watchdog has timed out.
        Returns True if fresh, False if stale.
        Triggers on_stale_callback upon transitioning to stale.
        """
        callback_to_call: Optional[Callable[[], None]] = None
        is_fresh = True
        with self._lock:
            now = time.time()
            elapsed = now - self._last_kick
            if elapsed > self.timeout_s:
                is_fresh = False
                if not self._is_stale:
                    self._is_stale = True
                    logger.warning(
                        f"Software watchdog TIMEOUT ({elapsed:.2f}s > {self.timeout_s:.2f}s). "
                        "Triggering SAFE_IDLE / STOP transition."
                    )
                    callback_to_call = self.on_stale_callback

        if callback_to_call:
            try:
                callback_to_call()
            except Exception as e:
                logger.error(f"Error in watchdog on_stale_callback: {e}", exc_info=True)

        return is_fresh

    def _monitor_loop(self):
        """Internal background loop checking watchdog status."""
        while self._running:
            self.check()
            time.sleep(self.check_interval_s)

    def start(self):
        """Starts background watchdog monitoring thread."""
        with self._lock:
            if self._running:
                return
            self._running = True
            self._last_kick = time.time()
            self._is_stale = False
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="SoftwareWatchdogMonitor",
                daemon=True,
            )
            self._thread.start()
            logger.info(f"Software watchdog started (timeout: {self.timeout_s}s)")

    def stop(self):
        """Stops background watchdog monitoring thread."""
        with self._lock:
            self._running = False
            thread = self._thread
            self._thread = None

        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        logger.info("Software watchdog stopped.")

    def reset(self):
        """Resets the watchdog state to fresh."""
        with self._lock:
            self._last_kick = time.time()
            self._is_stale = False
