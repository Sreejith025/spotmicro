# SpotMicro Quadruped Robot

Autonomous navigation, AI visual perception, inverse kinematics locomotion, and real-time telemetry dashboard for the SpotMicro quadruped robot platform.

---

## 1. Overview

**SpotMicro** is an open-source, 3D-printed quadruped robot platform powered by a Raspberry Pi single-board computer, 12 high-torque digital servos, multi-sensor telemetry, YOLO object detection, and a reactive navigation decision engine. 

The software architecture is strictly decoupled into high-level visual perception / navigation planning and low-level deterministic motor control, ensuring safety, modularity, and reproducible behavior both in real hardware and software simulation.

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

---

## 2. Core Architectural Philosophy & Safety Principles

1. **No Direct Joint or PWM Control from AI**: The AI perception model and navigation engine **never** emit raw PWM duty cycles or joint angles. Neural networks lack spatial and mechanical joint constraints; direct control causes motor chatter, leg collisions, and gear stripping.
2. **High-Level Discrete Commands Only**: The navigation engine produces only validated discrete high-level movement commands:
   - `FORWARD`
   - `BACKWARD`
   - `TURN_LEFT`
   - `TURN_RIGHT`
   - `STOP`
   - `AVOID_OBSTACLE`
   - `WAIT`
3. **Deterministic Locomotion (Inverse Kinematics)**: The `GaitController` deterministically translates high-level commands into smooth leg trajectories (trot / crawl gait) using 3-DOF analytical inverse kinematics.
4. **Hardware Driver Isolation**: `ServoController` is the sole authority communicating with the PCA9685 I2C driver, enforcing angle limits, calibration offsets, and speed ramping.
5. **Strict Safety Priority Hierarchy**:
   $$\text{Emergency Stop} \longrightarrow \text{Tip-over / IMU Tilt} \longrightarrow \text{Person Stop} \longrightarrow \text{Stairs / Cliff Hazard} \longrightarrow \text{Rule 9 Sensor Failure} \longrightarrow \text{Obstacle Avoidance} \longrightarrow \text{Autonomous Movement}$$
6. **Simulation First**: Software tests and dashboard simulations run in an isolated environment without activating physical servos, GPIO, or I2C buses.

---

## 3. Repository Structure

```text
d:\spotmicro\
├── README.md                      # Master repository documentation (this file)
├── train_spotmicro.py             # YOLO11n object detection training script
├── yolo11n.pt                     # Base pretrained YOLO11n weights
│
├── app/                           # Core autonomy & locomotion logic
│   ├── navigation.py              # NavigationDecisionEngine, YOLOPerceptionAdapter, Rule 9
│   ├── gait_controller.py         # Analytical IK, leg trajectory & gait coordination
│   ├── camera_pipeline.py         # Multi-threaded camera capture & YOLO inference HUD
│   ├── server.py                  # Standalone API server
│   └── NAVIGATION_README.md       # Detailed decision system & safety rules reference
│
├── hardware/                      # Low-level hardware drivers & HAL
│   ├── servo_controller.py        # PCA9685 16-channel 12-bit PWM driver
│   ├── imu.py                     # BNO055 9-DOF IMU interface & tilt detection
│   └── config.py                  # Pinouts, calibration offsets & joint angle boundaries
│
├── dataset/                       # Computer vision training data & roadmap
│   ├── README.md                  # Dataset roadmap & annotations guide
│   ├── detection/                 # Detection datasets and split configurations
│   └── detection_final_stage/     # Curated 6-class YOLO training dataset
│
├── web/                           # SpotMicro Web Dashboard (spotmicro-web)
│   ├── README.md                  # Dashboard documentation & deployment instructions
│   ├── backend/                   # Python Flask-SocketIO simulation backend
│   │   ├── server.py              # REST API & WebSocket server
│   │   ├── simulation.py          # Real-time quadruped telemetry simulator
│   │   └── navigation.py          # Standalone navigation decision engine
│   └── frontend/                  # React + TypeScript + Vite + Tailwind CSS + Leaflet
│       ├── package.json
│       ├── vite.config.ts         # Reverse proxy configuration
│       └── src/
│           ├── pages/             # Overview, Manual Control, AI Vision, Telemetry, Logs
│           ├── components/        # DestinationMap (Leaflet), D-Pad, HUD cards
│           └── services/          # WebSocket, REST API, SimulatedPositionProvider
│
├── tests/                         # Unit & integration test suites
│   ├── test_navigation.py         # Navigation decision engine & safety hierarchy tests
│   ├── test_camera_pipeline.py    # Camera pipeline & frame processing tests
│   ├── test_servos.py             # Servo angles, IK limits & PCA9685 driver tests
│   └── test_web_backend.py        # Web backend REST, WebSocket & Destination Auto tests
│
├── runs/                          # Model training runs & validation metrics
└── weights/                       # Exported model weights (best.pt, ONNX)
```

---

## 4. Hardware Specifications

| Component | Specification | Description |
| :--- | :--- | :--- |
| **Main SBC** | Raspberry Pi 5 (4GB / 8GB) | High-level decision engine, camera streaming & web server |
| **Servo Actuators** | 12 × DS3225 25kg·cm | High-torque digital metal-gear servos (3 DOF per leg: Coxa, Femur, Tibia) |
| **PWM Controller** | PCA9685 (I2C address `0x40`) | 16-channel 12-bit PWM driver running at 50 Hz |
| **IMU Sensor** | Bosch BNO055 (I2C) | 9-DOF absolute orientation (roll, pitch, yaw) with hardware sensor fusion |
| **Distance Sensors**| 4 × HC-SR04 | Ultrasonic rangefinders (Front, Left, Right, Rear) |
| **Camera** | Wide-Angle CSI / USB Camera | 640×480 @ 30 FPS for YOLO11 visual obstacle & hazard detection |
| **Power Supply** | 2S / 3S LiPo + Dual BEC | Isolated 5V rail for Raspberry Pi; high-current 6V rail for servos |

---

## 5. Software Features & Capabilities

### 5.1 AI Perception & YOLO11n
- 6-Class Detection Model: `person`, `obstacle`, `chair`, `stairs`, `door`, `robot`.
- Person detected in path $\to$ immediate safety **STOP**.
- Stairs hazard detected $\to$ immediate cliff edge **STOP**.
- Lateral corridor evaluation for collision avoidance steering.

### 5.2 Navigation Decision Engine & Safety Filter
- **Hysteresis & Oscillation Damping**: Turn cooldowns and direction lock prevent steering oscillations when traversing narrow corridors.
- **Dead-Man Safety Timeout**: Locomotion halts automatically if perception updates stop arriving within $0.5\,\text{s}$.
- **Rule 9 Sensor Integrity Check**: Invalid, non-finite, out-of-range, or stale sensor data triggers `WAIT` or `STOP`, never blind forward progression.
- **Dynamic Posture & Tilt Protection**: Real-time roll/pitch monitoring immediately halts movement if tilt exceeds safety thresholds ($\pm 20^\circ$).

### 5.3 Autonomous Navigation Modes
- **Free Auto Mode**: Continuous obstacle avoidance and autonomous room exploration.
- **Destination Auto Mode**:
  - Interactive **OpenStreetMap + Leaflet** interface.
  - **1.5 km Circular Operating Radius**: Radial boundary ($\sqrt{x^2 + y^2} \le 1500\,\text{m}$) centered at simulation origin (`12.971587, 77.594784`).
  - **Deterministic Waypoint Routing**: Modular waypoint generator with $100\,\text{m}–300\,\text{m}$ spacing for long routes and $2\,\text{m}$ spacing for local navigation.
  - **GPS Provider Abstraction**: Seamless separation between `SimulatedPositionProvider` and future physical `NEO6MPositionProvider`.
  - **Mutual Mode Exclusion**: Free Auto, Destination Auto, and Manual Control are strictly isolated with authority handoff locks.

### 5.4 Control Dashboard
- Real-time bi-directional telemetry over WebSockets (attitude, battery, ultrasonic map, camera HUD).
- Industrial dark robotics UI built with React, TypeScript, Vite, and Tailwind CSS.
- One-click Emergency Stop with hotkey support (<kbd>ESC</kbd> / <kbd>Space</kbd>).

---

## 6. Getting Started & Installation

### 6.1 Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 18.0 or higher with npm
- **Git**

### 6.2 Python Virtual Environment Setup
From the repository root (`D:\spotmicro`):

```bash
# Create and activate virtual environment
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Linux / Raspberry Pi:
source .venv/bin/activate

# Install dependencies
pip install -r web/requirements.txt
pip install ultralytics torch torchvision
```

### 6.3 Frontend Dashboard Setup
From `web/frontend`:

```bash
cd web/frontend
npm install
```

---

## 7. Running the Dashboard

### Terminal 1: Backend Server
From repository root (`D:\spotmicro`):
```bash
python -m web.backend.server
```
*Starts Flask-SocketIO simulation server at `http://127.0.0.1:5000`.*

### Terminal 2: Frontend Client
From `web/frontend`:
```bash
npm run dev
```
*Starts Vite development server at `http://localhost:5173`.*

Open your browser at **`http://localhost:5173`**.

---

## 8. Running Automated Tests

The repository includes comprehensive unit and integration test suites covering the navigation decision engine, camera pipeline, servo kinematics, and web backend:

```bash
# 1. Run Web Backend & Destination Auto tests
python -m unittest tests.test_web_backend

# 2. Run Navigation Decision Engine tests
python -m unittest tests.test_navigation

# 3. Run Camera Pipeline tests
python -m unittest tests.test_camera_pipeline

# 4. Run full combined test suite
python -m unittest tests.test_navigation tests.test_camera_pipeline tests.test_web_backend

# 5. Verify Frontend TypeScript production build
npm --prefix web/frontend run build
```

---

## 9. YOLO Training Pipeline

To train or fine-tune the 6-class YOLO11n object detection model on custom robot camera data:

```bash
python train_spotmicro.py
```
- Automatically verifies CUDA GPU availability and VRAM limits.
- Uses `dataset/detection_final_stage/data.yaml`.
- Saves weights and metrics to `runs/detect/spotmicro_yolo11n_v1/`.

---

## 10. Hardware Bench Verification Checklist

Before placing the robot on the floor or enabling physical servo power:

1. **Bench Stand**: Suspend robot chassis so all four legs move freely in the air without touching the ground.
2. **Servo Zero Calibration**: Verify individual joint 90° calibration angles in `hardware/config.py`.
3. **Emergency Stop Test**: Confirm physical or web UI Emergency Stop cuts PWM signals within $\le 50\,\text{ms}$.
4. **IMU Orientation**: Validate roll/pitch angles respond correctly when manually tilting the robot chassis.
5. **Sonar Range Check**: Verify ultrasonic distances read accurately from $2\,\text{cm}$ to $400\,\text{cm}$.

---

## 11. License & Acknowledgments

- SpotMicro mechanical design derived from the open-source SpotMicro community.
- Perception pipeline built with Ultralytics YOLO11.
- Dashboard built with React, Vite, Leaflet, and Tailwind CSS.

