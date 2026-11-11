"""
Unit Tests for SpotMicro Live Camera Perception Pipeline (Stage 2A)
===================================================================
Tests frame capture threading, synthetic camera generator, YOLO inference
adaptation, failure handling (camera disconnect, model exception), frame
annotation, and safe pipeline shutdown.

All tests run in software-only mode with mocked hardware and models.
"""

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.camera_pipeline import (
    CameraPipelineConfig,
    LiveCameraPipeline,
    OpenCVCameraCapture,
    SyntheticCameraCapture,
    parse_args,
)
from app.navigation import (
    BoundingBox,
    Detection,
    NavCommand,
    NavigationConfig,
    NavigationDecision,
    NavigationDecisionEngine,
    UltrasonicReading,
    UltrasonicSensors,
)


class TestSyntheticCameraCapture(unittest.TestCase):
    """Tests for the software-only synthetic camera frame generator."""

    def setUp(self):
        self.cam = SyntheticCameraCapture(width=320, height=240, scenario="clear")

    def test_start_and_read_frame(self):
        self.assertTrue(self.cam.start())
        self.assertTrue(self.cam.is_opened())

        success, frame, ts, err = self.cam.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertIsNone(err)
        self.assertGreater(ts, 0.0)

    def test_scenario_switching(self):
        self.cam.start()
        self.cam.set_scenario("person")
        success, frame_person, _, _ = self.cam.read_frame()
        self.assertTrue(success)

        self.cam.set_scenario("obstacle")
        success, frame_obs, _, _ = self.cam.read_frame()
        self.assertTrue(success)
        # Frames from different scenarios should differ
        self.assertFalse(np.array_equal(frame_person, frame_obs))

    def test_stop_release(self):
        self.cam.start()
        self.cam.stop()
        self.assertFalse(self.cam.is_opened())
        success, frame, _, err = self.cam.read_frame()
        self.assertFalse(success)
        self.assertIsNone(frame)
        self.assertIn("stopped", err.lower())


class TestOpenCVThreadedCapture(unittest.TestCase):
    """Tests for OpenCVCameraCapture threading, buffering, and error handling."""

    @patch("cv2.VideoCapture")
    def test_threaded_capture_mocked_success(self, mock_cv_class):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, dummy_frame)
        mock_cv_class.return_value = mock_cap

        cam = OpenCVCameraCapture(source=0, width=640, height=480)
        started = cam.start()
        self.assertTrue(started)
        self.assertTrue(cam.is_opened())

        # Allow background thread to poll at least once
        time.sleep(0.08)

        success, frame, ts, err = cam.read_frame()
        self.assertTrue(success)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertIsNone(err)
        self.assertGreater(ts, 0.0)

        cam.stop()
        self.assertFalse(cam.is_opened())
        mock_cap.release.assert_called_once()

    @patch("cv2.VideoCapture")
    def test_camera_failed_to_open(self, mock_cv_class):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv_class.return_value = mock_cap

        cam = OpenCVCameraCapture(source=99)
        started = cam.start()
        self.assertFalse(started)
        self.assertFalse(cam.is_opened())

        success, frame, _, err = cam.read_frame()
        self.assertFalse(success)
        self.assertIsNone(frame)
        self.assertIn("failed to open", err.lower())


class TestLiveCameraPipeline(unittest.TestCase):
    """Tests for LiveCameraPipeline perception orchestration and error recovery."""

    def setUp(self):
        self.mock_capture = MagicMock(spec=SyntheticCameraCapture)
        self.mock_capture.is_opened.return_value = True

        # Mock YOLO model
        self.mock_model = MagicMock()
        self.mock_model.names = {0: "person", 1: "obstacle", 2: "chair", 3: "stairs", 4: "door", 5: "robot"}

        # Custom nav config with 0 stop-hold frames for deterministic unit testing
        self.nav_config = NavigationConfig(
            consecutive_frames_required=1,
            stop_hold_frames=0,
            require_front_ultrasonic=True,
        )
        self.nav_engine = NavigationDecisionEngine(self.nav_config)

        self.pipeline_config = CameraPipelineConfig(
            camera_source=0,
            imgsz=320,
            conf_threshold=0.35,
            display=False,
            headless=True,
            log_decisions=False,
        )

        self.pipeline = LiveCameraPipeline(
            config=self.pipeline_config,
            nav_engine=self.nav_engine,
            model=self.mock_model,
            capture_device=self.mock_capture,
        )

    def test_camera_capture_failure_handling(self):
        """When camera capture fails, step() must fail safely to STOP/WAIT."""
        self.mock_capture.read_frame.return_value = (False, None, 0.0, "Device disconnected")

        decision = self.pipeline.step()
        self.assertIsNotNone(decision)
        # Missing sensors & no camera frame must yield WAIT or STOP, never FORWARD
        self.assertIn(decision.command, (NavCommand.WAIT, NavCommand.STOP))

    def test_model_inference_exception_handling(self):
        """When model inference raises an exception, pipeline must emit emergency STOP."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_model.side_effect = RuntimeError("CUDA Out of Memory in Conv2D")

        decision, detections, latency = self.pipeline.process_frame(dummy_frame, timestamp=time.time())
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("inference exception", decision.reason.lower())
        self.assertEqual(len(detections), 0)
        self.assertIn("inference_error", decision.telemetry)

    def test_person_blocking_path_triggers_stop(self):
        """When YOLO detects a person in center corridor, decision must be emergency STOP."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock YOLO results containing a person in center corridor
        class MockBoxes:
            def __init__(self):
                self.xyxyn = [[0.35, 0.2, 0.65, 0.85]]
                self.conf = [0.88]
                self.cls = [0]  # person

            def __len__(self):
                return len(self.cls)

        class MockResult:
            def __init__(self):
                self.boxes = MockBoxes()
                self.names = {0: "person", 1: "obstacle"}

        self.mock_model.side_effect = None
        self.mock_model.return_value = [MockResult()]

        decision, detections, latency = self.pipeline.process_frame(dummy_frame, timestamp=time.time())
        self.assertEqual(decision.command, NavCommand.STOP)
        self.assertIn("person", decision.reason.lower())
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].class_name, "person")

    def test_clear_frame_sensor_integrity(self):
        """When camera is clear, engine must observe Rule 9 (missing ultrasonic -> WAIT)."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        class EmptyBoxes:
            def __init__(self):
                self.xyxyn = []
                self.conf = []
                self.cls = []

            def __len__(self):
                return 0

        class MockResult:
            def __init__(self):
                self.boxes = EmptyBoxes()
                self.names = {0: "person", 1: "obstacle"}

        self.mock_model.side_effect = None
        self.mock_model.return_value = [MockResult()]

        # Without ultrasonic sensors (default), cannot assume clear path -> WAIT
        decision, detections, _ = self.pipeline.process_frame(dummy_frame, timestamp=time.time())
        self.assertEqual(decision.command, NavCommand.WAIT)
        self.assertIn("front ultrasonic reading is missing", decision.reason.lower())

        # If front ultrasonic reading is provided via telemetry override, path is clear -> FORWARD
        sensors = UltrasonicSensors(
            front=UltrasonicReading(distance_cm=120.0, timestamp=time.time())
        )
        decision_clear, _, _ = self.pipeline.process_frame(dummy_frame, timestamp=time.time(), ultrasonic_override=sensors)
        self.assertEqual(decision_clear.command, NavCommand.FORWARD)

    def test_annotate_frame_rendering(self):
        """Test HUD annotation rendering with bounding boxes and decision badge."""
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            Detection(class_name="person", confidence=0.85, bbox=BoundingBox(0.4, 0.2, 0.6, 0.8)),
            Detection(class_name="obstacle", confidence=0.72, bbox=BoundingBox(0.7, 0.5, 0.9, 0.8)),
        ]
        decision = NavigationDecision(command=NavCommand.STOP, reason="Person in center path")

        annotated = self.pipeline.annotate_frame(
            dummy_frame,
            detections=detections,
            decision=decision,
            latency_ms=45.2,
            fps=22.1
        )

        self.assertIsNotNone(annotated)
        self.assertEqual(annotated.shape, (480, 640, 3))
        self.assertEqual(annotated.dtype, np.uint8)
        # Annotated frame should have modified pixel values (HUD drawn)
        self.assertFalse(np.array_equal(annotated, dummy_frame))

    def test_safe_pipeline_shutdown(self):
        """Test pipeline start and clean shutdown."""
        self.pipeline.start()
        self.assertTrue(self.pipeline._running)
        self.pipeline.stop()
        self.assertFalse(self.pipeline._running)
        self.mock_capture.stop.assert_called_once()

    def test_missing_weights_file_raises_informative_error(self):
        """Verify non-existent weights path raises FileNotFoundError with SCP transfer guidance."""
        bad_config = CameraPipelineConfig(weights_path="/non/existent/path/best.pt")
        with self.assertRaises(FileNotFoundError) as ctx:
            LiveCameraPipeline(config=bad_config, nav_engine=self.nav_engine, model=None)
        self.assertIn("scp", str(ctx.exception).lower())


class TestCLIArgumentParsing(unittest.TestCase):
    """Tests for command-line interface arguments."""

    def test_parse_args_defaults(self):
        args = parse_args([])
        self.assertEqual(args.camera, "0")
        self.assertEqual(args.imgsz, 640)
        self.assertEqual(args.conf, 0.35)
        self.assertFalse(args.display)
        self.assertFalse(args.test_mode)
        self.assertIsNone(args.max_frames)

    def test_parse_args_custom(self):
        custom = [
            "--camera", "1",
            "--weights", "custom/best.pt",
            "--imgsz", "320",
            "--conf", "0.50",
            "--display",
            "--headless",
            "--test-mode",
            "--max-frames", "10",
        ]
        args = parse_args(custom)
        self.assertEqual(args.camera, "1")
        self.assertEqual(args.weights, "custom/best.pt")
        self.assertEqual(args.imgsz, 320)
        self.assertEqual(args.conf, 0.50)
        self.assertTrue(args.display)
        self.assertTrue(args.headless)
        self.assertTrue(args.test_mode)
        self.assertEqual(args.max_frames, 10)


if __name__ == "__main__":
    unittest.main()
