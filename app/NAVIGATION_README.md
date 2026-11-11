# SpotMicro Autonomous Navigation Decision System (Stage 1)

This document describes the design, architecture, safety principles, and verification results for the high-level **Navigation Decision Engine** of the SpotMicro quadruped robot.

---

## 1. System Architecture & Boundaries

The SpotMicro autonomy stack is built with a strict separation between high-level perception / decision-making and low-level deterministic motor control:

```text
 ┌─────────────────────────────────────────────────────────────┐
 │    Hardware Perception: Wide-Angle Camera, IMU, Ultrasonic  │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Raw Telemetry & Video Frames
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │       YOLO11n Object Detection & Perception Adapter         │
 │          (app/navigation.py: YOLOPerceptionAdapter)         │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Normalized Detections & Sensor Readings
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │             Navigation Decision Engine                      │
 │       (app/navigation.py: NavigationDecisionEngine)         │
 │   State Machine / Safety Filters / Hysteresis / Dead-Man   │
 └──────────────────────────────┬──────────────────────────────┘
                                │ High-Level NavCommand (FORWARD, STOP, etc.)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │          NavigationGaitBridge & GaitController              │
 │                 (app/gait_controller.py)                    │
 │             Inverse Kinematics & Leg Trajectories           │
 └──────────────────────────────┬──────────────────────────────┘
                                │ Target Joint Angles
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │              PCA9685 16-Channel Servo Driver                │
 │               (hardware/servo_controller.py)                │
 └──────────────────────────────┬──────────────────────────────┘
                                │ 50 Hz PWM Pulses (500–2500 µs)
                                ▼
 ┌─────────────────────────────────────────────────────────────┐
 │            12 × DS3225 High-Torque Digital Servos           │
 └─────────────────────────────────────────────────────────────┘
```

### Architectural Safety Boundaries
1. **No Direct Joint or PWM Control**: The AI perception model and navigation engine NEVER emit raw PWM pulses, duty cycles, or joint angles. Direct AI servo control introduces motor chatter, leg collisions, and instability.
2. **High-Level Discrete Commands Only**: The navigation engine produces only validated discrete high-level commands:
   - `FORWARD`
   - `BACKWARD`
   - `TURN_LEFT`
   - `TURN_RIGHT`
   - `STOP`
   - `AVOID_OBSTACLE`
   - `WAIT`
3. **Deterministic Gait Generation**: `GaitController` (`app/gait_controller.py`) deterministically manages gait timing (tripod/trot) and inverse kinematics.
4. **Hardware Driver Isolation**: `ServoController` (`hardware/servo_controller.py`) is the single authority communicating with the PCA9685 I2C driver.
5. **No Automatic Hardware Initialization**: Software tests and offline simulations run in an isolated environment without activating physical servos, GPIO, or I2C buses.

---

## 2. Input and Output Formats

### 2.1 Navigation Input (`NavigationInput`)
A structured packet representing synchronized sensor telemetry for one decision cycle:

```python
NavigationInput(
    detections: List[Detection],        # Bounding boxes, classes, confidence
    ultrasonic: UltrasonicSensors,      # Front, left, right, rear distances
    imu: Optional[IMUReading],          # Roll, pitch, yaw, stability status
    timestamp: float                    # Epoch seconds
)
```

#### Object Detections (`Detection`)
```python
Detection(
    class_name: str,       # 'person', 'obstacle', 'chair', 'stairs', 'door', 'robot'
    confidence: float,     # Bounded in [0.0, 1.0]
    bbox: BoundingBox      # Normalized coords: x_min, y_min, x_max, y_max in [0.0, 1.0]
)
```

#### Ultrasonic Sensors (`UltrasonicSensors`)
Contains readings for the four planned HC-SR04 sensors:
- `front`: Front proximity and path clearance (`distance_cm`, `timestamp`, `is_valid`)
- `left`: Lateral left clearance
- `right`: Lateral right clearance
- `rear`: Rear clearance (for future reverse validation)

Valid distance range: $2.0\text{ cm} \le d \le 400.0\text{ cm}$. Any reading outside this range, non-finite, negative, or older than `max_sensor_age_s` ($0.5\text{ s}$) is rejected as invalid/stale.

#### IMU Reading (`IMUReading`)
```python
IMUReading(
    roll: float,           # Side-to-side tilt in degrees (±20.0° limit)
    pitch: float,          # Nose up/down tilt in degrees (±20.0° limit)
    yaw: float,            # Heading in degrees
    is_stable: bool,       # Dynamic vibration / tipping hazard flag
    timestamp: float       # Epoch seconds
)
```

### 2.2 Navigation Decision (`NavigationDecision`)
The engine returns an immutable decision structure:

```python
NavigationDecision(
    command: NavCommand,       # High-level enum (e.g., NavCommand.FORWARD)
    reason: str,               # Human-readable rationale for debugging
    timestamp: float,          # Time decision was emitted
    sub_action: Optional[str], # Evasive sub-action: 'TURN_LEFT' or 'TURN_RIGHT'
    telemetry: Dict[str, Any]  # Telemetry snapshot (filters, sensor statuses)
)
```

---

## 3. Command Decision Rules & Priority Hierarchy

The decision logic evaluates rules in strict order of descending priority:

```text
  1. Dead-Man Policy / Pipeline Stall Check
     └─ If pipeline interval > deadman_timeout_s (1.0 s) ───────► STOP
  
  2. IMU Stability & Tilt Safety
     ├─ If dynamic instability reported ───────────────────────► Emergency STOP
     ├─ If |roll| > 20.0° or |pitch| > 20.0° ──────────────────► Emergency STOP
     └─ If IMU telemetry is stale ─────────────────────────────► Emergency STOP

  3. Person in Intended Path
     └─ If 'person' detected in center corridor (conf ≥ 0.40) ─► Emergency STOP

  4. Stairs Hazard Check
     └─ If 'stairs' detected ahead ────────────────────────────► Emergency STOP
        (Automatic stair climbing is strictly disabled)

  5. Sensor Integrity & Proximity Check
     ├─ If front ultrasonic is missing, stale, or invalid ─────► WAIT
     │  (Rule 9: Missing sensor data != clear path)
     └─ If front distance ≤ front_stop_distance_cm (30 cm) ────► Emergency STOP

  6. Center Corridor Obstruction Evaluation
     ├─ Front distance < caution threshold (65 cm), OR
     ├─ YOLO detects 'obstacle', 'chair', or 'robot' in center corridor, OR
     └─ YOLO detects 'door' without verified forward clearance:
        ├─ Evaluate left & right lateral clearances
        ├─ Left clear & right blocked ─────────────────────────► AVOID_OBSTACLE (TURN_LEFT)
        ├─ Right clear & left blocked ─────────────────────────► AVOID_OBSTACLE (TURN_RIGHT)
        ├─ Both sides clear ───────────────────────────────────► AVOID_OBSTACLE (Steer toward greater clearance)
        └─ Both sides blocked / unverified ────────────────────► STOP

  7. All Checks Clear
     └─ No obstacles detected, front clearance verified ───────► FORWARD
```

---

## 4. Safety Behaviors

### Rule 9 Compliance: Conservative Failure Modes
Missing, stale, or out-of-range sensor readings are **never** interpreted as proof that a path is clear:
- If the front ultrasonic sensor disconnects or fails, the engine issues `WAIT` rather than advancing blindly.
- If lateral sensors are absent during obstacle evasion, the robot will **not** assume the side is safe and instead halts.

### Emergency STOP Priority & Zero-Latency Bypass
Emergency stops (such as detecting a person, tipping, or proximity violations) immediately bypass hysteresis filters. The stop takes effect with 0 delay.

### Hysteresis & Consecutive-Frame Confirmation
To prevent rapid command switching ("fluttering") caused by noisy perception frames:
- Non-emergency transitions between moving commands (e.g., `FORWARD` $\to$ `AVOID_OBSTACLE`) require $N$ consecutive candidate frames (default: `2`).
- After a `STOP` command, the robot holds position for `stop_hold_frames` (default: `2`) before resuming locomotion to ensure the quadruped body has settled.

### Dead-Man Switch
If the camera loop, ROS node, or sensor telemetry thread stalls for longer than `deadman_timeout_s` ($1.0\text{ s}$), the engine immediately transitions to `STOP`.

### Safe Motion Constraints
- **Reverse Movement**: Unverified reverse movement is blocked by default (`allow_unverified_reverse = False`).
- **Stair Climbing**: The quadruped is not equipped with specialized stair climbing gaits. Stair detection triggers a safe `STOP`.
- **Doorways**: A detected doorway is not assumed to be an open passage. Clearance depth must be independently verified by distance sensing.

---

## 5. Perception Adapter (`YOLOPerceptionAdapter`)

The perception adapter bridges Ultralytics YOLO models to the decision engine:
- Reads either an Ultralytics `Results` instance (with `.boxes`) or a dictionary list.
- Automatically maps class IDs ($0\dots 5$) to names:
  ```python
  0: 'person'
  1: 'obstacle'
  2: 'chair'
  3: 'stairs'
  4: 'door'
  5: 'robot'
  ```
- Normalizes coordinates into $[0.0, 1.0]$.
- Rejects malformed bounding boxes ($x_{\min} \ge x_{\max}$, $y_{\min} \ge y_{\max}$, NaN, Inf, out-of-bounds).

---

## 6. Known Model Limitations

Our YOLO11n model (`D:\spotmicro\runs\detect\spotmicro_yolo11n_v1\weights\best.pt`) was trained on RTX 3050 with specific domain constraints:

| Class | Known Limitation | Navigation Mitigation |
| :--- | :--- | :--- |
| `person` | Generalizes well from COCO | Critical corridor triggers immediate `STOP`. |
| `obstacle` | Trained on general indoor/outdoor obstacles | Paired with ultrasonic verification. |
| `chair` | Training accuracy is weak | Lowered confidence threshold ($0.30$) to detect chair legs early. |
| `stairs` | Insufficient training examples | Any stair detection triggers immediate `STOP` rather than an attempt to climb. |
| `door` | Recognizes door frames, not whether door is ajar | Does not grant traversability without ultrasonic depth verification. |
| `robot` | Zero real quadruped training examples | Included in class schema for future multi-robot coordination. |

---

## 7. Gait Controller Integration (`NavigationGaitBridge`)

The `NavigationGaitBridge` safely routes decisions to `GaitController` (`app/gait_controller.py`):

```python
from app.gait_controller import GaitController
from app.navigation import NavigationDecisionEngine, NavigationGaitBridge, NavCommand

# Initialize controllers
gait = GaitController()
engine = NavigationDecisionEngine()
bridge = NavigationGaitBridge(gait_controller=gait, allow_backward=False)

# During the execution loop:
decision = engine.decide(nav_input)
gait_command = bridge.dispatch(decision)
# gait_command will be one of: 'forward', 'left', 'right', 'stop'
```

Mapping:
- `NavCommand.FORWARD` $\to$ `"forward"`
- `NavCommand.STOP` $\to$ `"stop"`
- `NavCommand.WAIT` $\to$ `"stop"`
- `NavCommand.AVOID_OBSTACLE` (steer left) $\to$ `"left"`
- `NavCommand.AVOID_OBSTACLE` (steer right) $\to$ `"right"`
- `NavCommand.BACKWARD` $\to$ `"stop"` (safely blocked unless verified)

---

## 8. Running the Tests

The unit test suite runs in isolation without touching hardware drivers:

```bash
# Run unit tests directly
python tests/test_navigation.py -v

# Or run via unittest module
python -m unittest tests.test_navigation -v
```

All 36 unit tests cover:
- Bounding box boundary conditions & NaN/Inf safety
- Ultrasonic range limits ($2\text{ cm} \le d \le 400\text{ cm}$) and staleness
- IMU roll/pitch tilt limits and dynamic instability alerts
- Person in center path emergency STOP
- Obstacle in center path triggering avoidance
- Blocked left side, blocked right side, and both sides blocked
- Clear path forward progression
- Low-confidence filtering
- Missing & stale ultrasonic data handling (Rule 9)
- Dead-man switch timeout
- Automatic stairs refusal
- Doorway caution logic
- YOLO perception adapter (dictionary & Ultralytics Results parsing)
- Gait controller integration dispatch & reverse block

## 9. Stage 2A: Live Camera Pipeline & Software-Only Preview

The live camera perception pipeline is implemented in [app/camera_pipeline.py](file:///d:/spotmicro/app/camera_pipeline.py).

### Pipeline Features:
- **Threaded Capture**: Captures frames in a dedicated daemon thread to prevent frame backlog and stale frame latency.
- **YOLO11n Real-time Inference**: Processes camera frames through `best.pt` with configurable resolution (`--imgsz 640` or `320`) and confidence threshold (`--conf 0.35`).
- **Navigation Decision HUD**: Annotates video preview with central navigation corridor guide lines (`x in [0.25, 0.75]`), color-coded bounding boxes, high-contrast decision badges, and real-time latency/FPS.
- **Safe Failure Modes**: If the camera disconnects, read returns an empty frame, or inference raises an exception, the system safely triggers `STOP` / `WAIT`.
- **Rule 9 Enforcement**: Does not invent synthetic sensor values. Without physical ultrasonic sensors connected, path clearance is not assumed (`WAIT`).

### Running the Camera Preview (Software-Only / No Hardware Motion):

```bash
# 1. Development preview with GUI window and physical webcam:
python -m app.camera_pipeline --camera 0 --display

# 2. Software-only test mode (no physical camera required):
python -m app.camera_pipeline --test-mode --headless --max-frames 30

# 3. Headless mode on Raspberry Pi OS (over SSH):
python -m app.camera_pipeline --camera /dev/video0 --headless --imgsz 320
```

---

## 10. Raspberry Pi OS Deployment Guide & Model Transfer

### Model Storage & Memory Requirements:
- **Model Checkpoint Size**: `best.pt` is `5.21 MB` (`5,464,794 bytes`).
- **RAM Overhead**: ~400 MB RAM for PyTorch/YOLO CPU runtime.
- **Disk Storage**: ~1.5 GB on Raspberry Pi for PyTorch CPU + Ultralytics + OpenCV packages.

### Safe Model Transfer from Laptop to Pi:
On your laptop, transfer the trained weights file directly over SSH:
```bash
scp D:\spotmicro\runs\detect\spotmicro_yolo11n_v1\weights\best.pt pi@<RPI_IP>:~/spotmicro/runs/detect/spotmicro_yolo11n_v1/weights/best.pt
```

### Raspberry Pi OS Package Dependencies:
Run on Raspberry Pi 5 (Debian Bookworm 64-bit):
```bash
# System packages for camera & computer vision
sudo apt-get update
sudo apt-get install -y python3-pip python3-opencv libcamera-tools v4l-utils

# Python dependencies (in a virtual environment recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

---

## 11. Remaining Steps Before Physical Hardware Deployment

1. **Pi Camera Driver Validation**: Test `/dev/video0` or `libcamerify` on Raspberry Pi 5.
2. **Model Quantization Benchmark**: Benchmark PyTorch CPU vs. ONNX FP16 directly on the Raspberry Pi 5 to measure actual achievable FPS (do not claim 20 FPS until measured on real hardware).
3. **Ultrasonic Driver Implementation**: Write non-blocking GPIO readers for HC-SR04 sensors with $1\text{ k}\Omega / 2\text{ k}\Omega$ voltage dividers on echo lines.
4. **BNO055 IMU Driver Integration**: Connect I2C telemetry stream from `hardware/imu.py`.
5. **Elevated Stand Verification**: Test complete autonomy stack with quadruped elevated on a test stand (legs off the ground).
