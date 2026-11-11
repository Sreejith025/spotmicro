"""
SpotMicro Live Camera Integration & Perception Pipeline (Stage 2A)
==================================================================
Captures live camera frames in a background thread, runs YOLO11n object
detection, maps detections into structured navigation input, and evaluates
high-level navigation decisions with real-time visual and console telemetry.

SAFETY CONSTRAINTS:
- Software-only execution in this stage.
- Strictly NO physical servo, GPIO, I2C, PCA9685, or motor activation.
- Does NOT dispatch movement commands to a physical gait controller.
- If camera capture or model inference fails, fails safely (STOP/WAIT)
  and NEVER assumes the path is clear.
- Does NOT invent synthetic sensor values to force FORWARD.
- Frame capture uses a dedicated single-frame buffer to eliminate stale frame lag.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Project root resolution for cross-platform portability (Linux Raspberry Pi OS / Windows)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.navigation import (
    BoundingBox,
    Detection,
    NavCommand,
    NavigationConfig,
    NavigationDecision,
    NavigationDecisionEngine,
    NavigationInput,
    UltrasonicSensors,
    YOLOPerceptionAdapter,
)

# Configure logger
logger = logging.getLogger("SpotMicroCameraPipeline")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Default model path relative to project root
DEFAULT_WEIGHTS_PATH = PROJECT_ROOT / "runs" / "detect" / "spotmicro_yolo11n_v1" / "weights" / "best.pt"


# ---------------------------------------------------------------------------
# Camera Pipeline Configuration
# ---------------------------------------------------------------------------

@dataclass
class CameraPipelineConfig:
    """Configuration parameters for the live camera perception pipeline."""
    camera_source: Union[int, str] = 0           # Device index (0, 1), video file, or v4l2 device
    weights_path: Optional[Union[str, Path]] = None  # Path to YOLO model weights
    imgsz: int = 640                            # YOLO inference input dimension (e.g. 640 or 320)
    conf_threshold: float = 0.35                # Detection confidence filter threshold
    camera_width: int = 640                     # Camera frame width
    camera_height: int = 480                    # Camera frame height
    display: bool = False                       # Whether to render real-time OpenCV window
    headless: bool = False                      # Force headless operation (disable cv2.imshow)
    log_decisions: bool = True                  # Print navigation decisions to console
    test_mode: bool = False                     # Software-only synthetic test frame generator
    target_fps: float = 30.0                    # Desired processing loop rate cap

    def get_resolved_weights_path(self) -> Path:
        """Resolves weights path cross-platform, checking environment variable and defaults."""
        if self.weights_path is not None:
            return Path(self.weights_path)
        env_weights = os.environ.get("SPOTMICRO_WEIGHTS")
        if env_weights:
            return Path(env_weights)
        return DEFAULT_WEIGHTS_PATH


# ---------------------------------------------------------------------------
# Threaded Camera Capture
# ---------------------------------------------------------------------------

class OpenCVCameraCapture:
    """
    Threaded OpenCV VideoCapture to prevent frame lag and backlog.
    Continuously reads frames into a single-slot buffer so the perception
    loop always consumes the newest available frame.
    """

    def __init__(self, source: Union[int, str] = 0, width: int = 640, height: int = 480):
        self.source = source
        self.width = width
        self.height = height

        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._frame_count: int = 0
        self._dropped_frames: int = 0
        self._last_read_frame_id: int = -1
        self._error_status: Optional[str] = None

    def start(self) -> bool:
        """Opens camera and starts the background acquisition thread."""
        if self._running:
            return True

        logger.info(f"Opening camera source: {self.source} ({self.width}x{self.height})")
        # On Windows, try DSHOW first for fast startup; on Linux V4L2 is used automatically
        if isinstance(self.source, int) and sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap.isOpened():
            self._error_status = f"Failed to open camera source: {self.source}"
            logger.error(self._error_status)
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self._running = True
        self._thread = threading.Thread(target=self._capture_worker, name="CameraCaptureThread", daemon=True)
        self._thread.start()

        # Wait briefly for the first frame
        start_wait = time.time()
        while time.time() - start_wait < 1.0:
            with self._lock:
                if self._latest_frame is not None:
                    break
            time.sleep(0.02)

        return True

    def _capture_worker(self):
        """Continuously polls the camera in a background thread."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                with self._lock:
                    self._error_status = "Camera device disconnected or unopened"
                break

            ret, frame = self._cap.read()
            capture_time = time.time()

            if not ret or frame is None:
                with self._lock:
                    self._error_status = "Camera read returned empty frame (device lost or EOF)"
                time.sleep(0.05)
                continue

            with self._lock:
                if self._frame_count > 0 and self._last_read_frame_id < self._frame_count - 1:
                    self._dropped_frames += (self._frame_count - 1 - self._last_read_frame_id)
                self._latest_frame = frame
                self._latest_timestamp = capture_time
                self._frame_count += 1
                self._error_status = None

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float, Optional[str]]:
        """
        Retrieves the freshest captured frame.
        Returns: (success, frame_bgr, timestamp, error_message).
        """
        with self._lock:
            if self._error_status is not None and self._latest_frame is None:
                return False, None, 0.0, self._error_status
            if self._latest_frame is None:
                return False, None, 0.0, "No frame available yet"

            frame = self._latest_frame.copy()
            ts = self._latest_timestamp
            self._last_read_frame_id = self._frame_count - 1
            return True, frame, ts, None

    def is_opened(self) -> bool:
        with self._lock:
            return self._running and (self._cap is not None and self._cap.isOpened())

    def stop(self):
        """Stops the capture worker and releases the camera device."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._latest_frame = None
        logger.info(f"Camera capture stopped (grabbed: {self._frame_count}, dropped: {self._dropped_frames})")


# ---------------------------------------------------------------------------
# Synthetic Camera Capture for Software-Only Testing
# ---------------------------------------------------------------------------

class SyntheticCameraCapture:
    """
    Generates synthetic frames for unit tests and simulation without hardware.
    Supports injecting synthetic hazards (person, obstacle, clear corridor).
    """

    def __init__(self, width: int = 640, height: int = 480, scenario: str = "clear"):
        self.width = width
        self.height = height
        self.scenario = scenario
        self._running = False
        self._frame_count = 0

    def start(self) -> bool:
        self._running = True
        return True

    def set_scenario(self, scenario: str):
        self.scenario = scenario

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray], float, Optional[str]]:
        if not self._running:
            return False, None, 0.0, "Synthetic camera stopped"

        # Create a synthetic hallway image
        frame = np.full((self.height, self.width, 3), 40, dtype=np.uint8)
        cv2.line(frame, (0, self.height), (int(self.width * 0.3), int(self.height * 0.4)), (70, 70, 70), 2)
        cv2.line(frame, (self.width, self.height), (int(self.width * 0.7), int(self.height * 0.4)), (70, 70, 70), 2)

        # Draw scenario elements
        if self.scenario == "person":
            # Draw synthetic person silhouette in center corridor
            px1, py1 = int(self.width * 0.4), int(self.height * 0.25)
            px2, py2 = int(self.width * 0.6), int(self.height * 0.85)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (180, 180, 180), -1)
            cv2.circle(frame, (int(self.width * 0.5), int(self.height * 0.2)), 25, (180, 180, 180), -1)
        elif self.scenario == "obstacle":
            # Draw synthetic box obstacle
            ox1, py1 = int(self.width * 0.35), int(self.height * 0.45)
            ox2, py2 = int(self.width * 0.65), int(self.height * 0.8)
            cv2.rectangle(frame, (ox1, py1), (ox2, py2), (120, 140, 160), -1)

        self._frame_count += 1
        return True, frame, time.time(), None

    def is_opened(self) -> bool:
        return self._running

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# Live Camera Navigation Pipeline
# ---------------------------------------------------------------------------

class LiveCameraPipeline:
    """
    Main orchestration class integrating Live Camera, YOLO11n Inference,
    YOLOPerceptionAdapter, and NavigationDecisionEngine.
    """

    def __init__(
        self,
        config: Optional[CameraPipelineConfig] = None,
        nav_engine: Optional[NavigationDecisionEngine] = None,
        model: Optional[Any] = None,
        capture_device: Optional[Any] = None,
    ):
        self.config = config or CameraPipelineConfig()
        self.nav_engine = nav_engine or NavigationDecisionEngine()
        self.capture_device = capture_device
        self.model = model

        self._running = False
        self._processed_frames = 0
        self._last_decision: Optional[NavigationDecision] = None
        self._last_detections: List[Detection] = []
        self._fps_history: List[float] = []

        # Initialize model if not passed in
        if self.model is None:
            self._init_model()

        # Initialize camera source if not passed in
        if self.capture_device is None:
            if self.config.test_mode:
                self.capture_device = SyntheticCameraCapture(
                    width=self.config.camera_width,
                    height=self.config.camera_height
                )
            else:
                self.capture_device = OpenCVCameraCapture(
                    source=self.config.camera_source,
                    width=self.config.camera_width,
                    height=self.config.camera_height
                )

    def _init_model(self):
        """Loads the Ultralytics YOLO model from weights path."""
        weights_path = self.config.get_resolved_weights_path()
        if not weights_path.exists():
            error_msg = (
                f"Trained YOLO model weights not found at: {weights_path}\n"
                f"If deploying to Raspberry Pi, transfer the trained model from your laptop using SCP:\n"
                f"  scp D:\\spotmicro\\runs\\detect\\spotmicro_yolo11n_v1\\weights\\best.pt "
                f"pi@<RPI_IP>:~/spotmicro/runs/detect/spotmicro_yolo11n_v1/weights/best.pt"
            )
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        logger.info(f"Loading YOLO11n model from: {weights_path}")
        from ultralytics import YOLO
        self.model = YOLO(str(weights_path))
        logger.info(f"YOLO11n model loaded successfully. Classes: {getattr(self.model, 'names', {})}")

        # Warmup model to avoid first-frame kernel compilation lag
        try:
            dummy = np.zeros((self.config.imgsz, self.config.imgsz, 3), dtype=np.uint8)
            self.model(dummy, imgsz=self.config.imgsz, verbose=False)
            logger.info("YOLO11n model warmup completed.")
        except Exception as e:
            logger.warning(f"Model warmup skipped: {e}")

    def start(self) -> bool:
        """Starts the camera acquisition thread and pipeline."""
        if self._running:
            return True
        success = self.capture_device.start()
        if not success:
            logger.error("Failed to start camera capture device")
            return False
        self._running = True
        logger.info("Live camera navigation pipeline started successfully (Software-Only Mode)")
        return True

    def stop(self):
        """Stops the camera pipeline and releases OpenCV windows."""
        self._running = False
        if self.capture_device is not None:
            self.capture_device.stop()
        if self.config.display and not self.config.headless:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        logger.info(f"Live camera navigation pipeline stopped. Total frames processed: {self._processed_frames}")

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: float,
        ultrasonic_override: Optional[UltrasonicSensors] = None
    ) -> Tuple[NavigationDecision, List[Detection], float]:
        """
        Executes perception and navigation decision for a single frame.
        Returns: (decision, detections, inference_latency_ms)
        """
        t_start = time.perf_counter()

        # 1. Run YOLO inference safely
        try:
            results = self.model(
                frame,
                imgsz=self.config.imgsz,
                conf=self.config.conf_threshold,
                verbose=False
            )
        except Exception as e:
            logger.error(f"YOLO model inference failed: {e}")
            # Safe emergency STOP on inference crash
            emergency_decision = NavigationDecision(
                command=NavCommand.STOP,
                reason=f"Model inference exception: {e}",
                timestamp=timestamp,
                telemetry={"inference_error": str(e)}
            )
            return emergency_decision, [], 0.0

        # 2. Adapt detections using existing YOLOPerceptionAdapter
        detections = YOLOPerceptionAdapter.from_ultralytics_results(
            results,
            image_shape=frame.shape[:2]
        )

        # 3. Construct NavigationInput without forging sensor readings
        # Note: Ultrasonic readings are empty unless provided by caller/telemetry.
        sensors = ultrasonic_override if ultrasonic_override is not None else UltrasonicSensors()
        nav_input = NavigationInput(
            detections=detections,
            ultrasonic=sensors,
            imu=None,
            timestamp=timestamp
        )

        # 4. Query the deterministic NavigationDecisionEngine
        decision = self.nav_engine.decide(nav_input)
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        self._last_decision = decision
        self._last_detections = detections
        self._processed_frames += 1

        return decision, detections, latency_ms

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        decision: NavigationDecision,
        latency_ms: float,
        fps: float
    ) -> np.ndarray:
        """
        Renders HUD overlays: corridor guide lines, bounding boxes,
        and high-contrast decision status badge for development monitoring.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        # 1. Draw central navigation corridor guide lines
        x_min_corridor = int(w * self.nav_engine.config.corridor_x_min)
        x_max_corridor = int(w * self.nav_engine.config.corridor_x_max)
        cv2.line(annotated, (x_min_corridor, 0), (x_min_corridor, h), (0, 200, 255), 1, cv2.LINE_AA)
        cv2.line(annotated, (x_max_corridor, 0), (x_max_corridor, h), (0, 200, 255), 1, cv2.LINE_AA)

        # 2. Draw detection bounding boxes
        for det in detections:
            x1 = int(det.bbox.x_min * w)
            y1 = int(det.bbox.y_min * h)
            x2 = int(det.bbox.x_max * w)
            y2 = int(det.bbox.y_max * h)

            cls_name = det.class_name.lower()
            # Color coding: Person/Stairs -> Red; Obstacle/Chair -> Orange; Others -> Cyan
            if cls_name in ("person", "stairs"):
                color = (0, 0, 255)  # Red
            elif cls_name in ("obstacle", "chair"):
                color = (0, 140, 255)  # Orange
            else:
                color = (255, 200, 0)  # Cyan/Yellow

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, max(0, y1 - lh - 6)), (x1 + lw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, max(lh + 2, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Draw top decision banner
        cmd = decision.command.value
        # Badge color: STOP -> Red; FORWARD -> Green; AVOID/WAIT -> Yellow
        if cmd == "STOP":
            badge_color = (0, 0, 220)
        elif cmd == "FORWARD":
            badge_color = (0, 180, 0)
        else:
            badge_color = (0, 165, 255)

        cv2.rectangle(annotated, (0, 0), (w, 42), (20, 20, 20), -1)
        # Decision Command Badge
        cv2.rectangle(annotated, (8, 6), (150, 36), badge_color, -1)
        sub = f" ({decision.sub_action})" if decision.sub_action else ""
        cv2.putText(annotated, f"{cmd}{sub}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        # Decision Reason snippet
        reason_text = decision.reason[:65] + "..." if len(decision.reason) > 65 else decision.reason
        cv2.putText(annotated, reason_text, (160, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # 4. Draw FPS / Latency footer
        footer_text = f"Latency: {latency_ms:.1f}ms | FPS: {fps:.1f} | Detections: {len(detections)}"
        cv2.putText(annotated, footer_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        return annotated

    def step(self) -> Optional[NavigationDecision]:
        """
        Executes one step of capturing, inference, and display.
        Returns the NavigationDecision or None if capture is unavailable.
        """
        success, frame, ts, err = self.capture_device.read_frame()

        if not success or frame is None:
            # Handle camera capture failure safely
            logger.warning(f"Camera frame capture failed: {err}")
            fail_decision = self.nav_engine.decide(
                NavigationInput(detections=[], timestamp=time.time())
            )
            return fail_decision

        t_loop = time.perf_counter()
        decision, detections, latency_ms = self.process_frame(frame, ts)

        # Calculate FPS
        loop_time = time.perf_counter() - t_loop
        fps = 1.0 / max(loop_time, 0.001)
        self._fps_history.append(fps)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        avg_fps = sum(self._fps_history) / len(self._fps_history)

        # Console logging
        if self.config.log_decisions:
            sub = f" [{decision.sub_action}]" if decision.sub_action else ""
            det_summary = ", ".join(f"{d.class_name}({d.confidence:.2f})" for d in detections) or "none"
            logger.info(f"[{decision.command.value}{sub}] {decision.reason} | Detections: {det_summary} | {latency_ms:.1f}ms ({avg_fps:.1f} FPS)")

        # Render display window if enabled and not headless
        if self.config.display and not self.config.headless:
            try:
                annotated = self.annotate_frame(frame, detections, decision, latency_ms, avg_fps)
                cv2.imshow("SpotMicro Live Camera Navigation (Dev Preview)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("'q' pressed: exiting preview loop.")
                    self._running = False
            except cv2.error as e:
                logger.warning(f"Display rendering unavailable (headless environment?): {e}")
                self.config.headless = True

        return decision

    def run(self, max_frames: Optional[int] = None):
        """
        Executes continuous pipeline loop until interrupted or max_frames reached.
        """
        if not self.start():
            return

        self.nav_engine.reset()
        logger.info("Entering live perception loop. Press Ctrl+C or 'q' in preview window to exit.")
        frame_idx = 0

        try:
            while self._running:
                self.step()
                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    logger.info(f"Reached max requested frames ({max_frames}). Exiting loop.")
                    break
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received: shutting down pipeline.")
        finally:
            self.stop()


# ---------------------------------------------------------------------------
# CLI Argument Parser & Entry Point
# ---------------------------------------------------------------------------

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments for the live camera pipeline."""
    parser = argparse.ArgumentParser(
        description="SpotMicro Live Camera Navigation Pipeline (Software-Only Stage 2A)"
    )
    parser.add_argument(
        "--camera", "-c",
        default="0",
        help="Camera source: integer index (0, 1), video file path, or /dev/video0 (default: 0)"
    )
    parser.add_argument(
        "--weights", "-w",
        default=str(DEFAULT_WEIGHTS_PATH),
        help=f"Path to trained YOLO weights file (default: {DEFAULT_WEIGHTS_PATH})"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="YOLO inference resolution (e.g. 640, 320) (default: 640)"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Confidence threshold for object detection (default: 0.35)"
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Render graphical preview window with detection and decision HUD"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Force headless mode without graphical windows (recommended on Raspberry Pi SSH)"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Use synthetic software-only test frames instead of physical camera hardware"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Process specified number of frames then exit cleanly (useful for testing)"
    )

    return parser.parse_args(args)


def main():
    """Main CLI entry point for SpotMicro live camera navigation."""
    args = parse_args()

    # Parse camera source (integer if digit)
    cam_source: Union[int, str] = int(args.camera) if str(args.camera).isdigit() else args.camera

    config = CameraPipelineConfig(
        camera_source=cam_source,
        weights_path=args.weights,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        display=args.display,
        headless=args.headless,
        test_mode=args.test_mode
    )

    pipeline = LiveCameraPipeline(config=config)
    pipeline.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
