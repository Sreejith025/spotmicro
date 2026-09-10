# SpotMicro Autonomous Navigation Dataset Strategy

Master dataset strategy and roadmap for the **SpotMicro quadruped robot** equipped with Raspberry Pi 5, 12 x DS3225 servos (driven by PCA9685), BNO055 IMU, HC-SR04 ultrasonic sensors, and a wide-angle camera.

---

## 1. System Architecture & Control Flow

The AI perception and navigation stack is decoupled from low-level joint motor control to guarantee hardware safety and smooth gait execution.

> [!IMPORTANT]
> **Core Architectural Rule**: The AI system outputs **high-level directional actions** (`FORWARD`, `BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`, `AVOID_OBSTACLE`). The AI does **NOT** directly output servo PWM values or individual joint angles. Low-level gait generation and Inverse Kinematics (IK) are handled deterministically by `GaitController` and `ServoController`.

```text
       ┌────────────────────────────────────────────────────────┐
       │     Camera + BNO055 IMU + HC-SR04 Ultrasonic Sensors   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │             Perception & Vision Stack                  │
       │  (YOLO Object Detection + Monocular Depth Estimation)   │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │              Navigation Decision Engine                │
       │       (Outputs: FORWARD / TURN_LEFT / STOP / etc.)     │
       └───────────────────────────┬────────────────────────────┘
                                   │ (High-level Command)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │         app/gait_controller.py (GaitController)        │
       │      (Generates joint trajectories via Inverse Kinematics)│
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │       hardware/servo_controller.py (PCA9685 Driver)    │
       └───────────────────────────┬────────────────────────────┘
                                   │ (50 Hz PWM Pulses)
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │           12 x DS3225 High-Torque Servos               │
       └────────────────────────────────────────────────────────┘
```

---

## 2. End-to-End Recommended Dataset Pipeline

```text
                  [PUBLIC DATASETS]
        (COCO, Roboflow, RUGD, NYU Depth V2, etc.)
                          │
                          ▼
        [GENERAL PERCEPTION & NAVIGATION KNOWLEDGE]
            (Foundational feature representations)
                          │
                          ▼
              [SELECTED USEFUL SUBSETS]
            (Filtered classes, ≤15 GB storage)
                          │
                          ▼
       [OUR OWN COLLEGE INDOOR/OUTDOOR IMAGE DATA]
         (Hallways, stairwells, grass, pathways)
                          │
                          ▼
        [SPOTMICRO SENSOR + MOVEMENT TELEMETRY]
        (Camera + IMU + Servos + Ultrasonic logs)
                          │
                          ▼
               [OFFBOARD TRAINING / FINE-TUNING]
              (Laptop / Cloud GPU training)
                          │
                          ▼
       [VALIDATION ON UNSEEN COLLEGE ENVIRONMENTS]
         (Testing generalization & edge cases)
                          │
                          ▼
             [RASPBERRY PI 5 DEPLOYMENT]
           (Quantized ONNX/NCNN inference)
```

---

## 3. High-Priority vs. Excluded Datasets Summary

### 3.1 Recommended Datasets & Priority Matrix

| Dataset Name | Domain | Priority | Recommended Strategy | Purpose for SpotMicro |
| :--- | :--- | :---: | :--- | :--- |
| **COCO** | Perception | **HIGH** | **Subset Only** (~1.5 GB) | Baseline weights for `person` & `chair` detection |
| **Roboflow Navigation** | Detection | **HIGH** | **Subset Only** (~1.0 GB) | Pre-labeled `stairs`, `door`, `obstacle`, `robot` YOLO data |
| **NYU Depth V2** | Indoor Depth | **HIGH** | **Complete Set** (~2.8 GB) | Monocular indoor depth estimation pre-training |
| **RUGD** | Outdoor Navigation | **HIGH** | **Complete / Main Split** (~15 GB) | Outdoor path & ground surface traversability |
| **RELLIS-3D** | Outdoor Terrain | **HIGH** | **Subset (RGB+IMU)** (~12 GB) | Off-road rough terrain & obstacle segmentation |
| **TartanGround** | Ground Robotics | **HIGH** | **Subset Only** (~8 GB) | Ground-level camera perspective matching SpotMicro height |
| **ETH ANYmal Datasets** | Quadruped Dynamics | **HIGH** | **Subset Only** (~5 GB) | Quadruped IMU, stair climbing, & gait stability reference |
| **Open Images V7** | Perception | **MEDIUM** | **Subset Only** (~2.0 GB) | Target extraction for rare classes (`Stairs`, `Door`) |
| **TartanAir** | Synthetic Nav | **MEDIUM** | **Subset Only** (~5 GB) | Indoor/outdoor synthetic SLAM & depth baseline |
| **HM3D / HM3DSem** | Indoor Simulation | **MEDIUM** | **Rendered Frames** (~4 GB) | Indoor room-to-room navigation training |

---

### 3.2 What Should NOT Be Downloaded ❌

To prevent disk bloat and wasted bandwidth, the following datasets **MUST NOT** be downloaded in full:

1. ❌ **Full COCO Dataset (25 GB)**: Download only filtered images containing target classes (`person`, `chair`).
2. ❌ **Full Open Images V7 (>500 GB)**: NEVER download the complete archive. Download only specific class subsets using FiftyOne or Google scripts.
3. ❌ **Full TartanAir Dataset (3 TB)**: Extremely large; download only 1 or 2 small indoor/outdoor trajectory sequences (~5 GB).
4. ❌ **Raw HM3D 3D Mesh Assets (200 GB)**: Do NOT download raw 3D mesh files; only render required 2D RGB-D camera frames via Habitat-Sim.
5. ❌ **Pascal VOC 2012 (2 GB)**: Redundant with COCO and Roboflow; skipped to avoid duplication.
6. ❌ **Heavy 3D LiDAR Point Clouds from RELLIS-3D / Waymo (>100 GB)**: SpotMicro relies on RGB camera, IMU, and ultrasonic sensors; discard heavy 3D LiDAR files.

---

## 4. Concise 10-Phase Dataset & Navigation Roadmap

- **Phase 1: Public Dataset Selection**
  - Finalize target public dataset list (COCO, Roboflow, RUGD, NYU Depth V2, TartanGround, ANYmal).

- **Phase 2: Dataset Preparation & Subsetting**
  - Run subset filtering scripts to extract target classes and convert annotations into standard YOLO text format (`class_id x_center y_center width height`).

- **Phase 3: College Campus Indoor/Outdoor Image Collection**
  - Capture low-angle ($15\text{--}25\text{ cm}$) camera images across classrooms, corridors, stairwells, gravel, grass, and outdoor campus paths.

- **Phase 4: Manual Annotation & Verification**
  - Manually annotate custom college images using bounding boxes and verify all annotations before training.

- **Phase 5: Offboard Model Training**
  - Train YOLOv8n/v11n detection and monocular depth models on laptop/cloud GPU infrastructure (do NOT train on Raspberry Pi 5).

- **Phase 6: Cross-Validation on Unseen Environments**
  - Evaluate model precision, recall, and depth accuracy on unseen campus locations and lighting conditions.

- **Phase 7: SpotMicro Sensor Telemetry Collection**
  - Record synchronized telemetry logs on SpotMicro: RGB camera frames, BNO055 IMU orientation/acceleration, HC-SR04 ultrasonic distance, servo angles, and gait state (`forward`, `left`, `stop`).

- **Phase 8: High-Level Navigation Stack Integration**
  - Integrate perception outputs into the high-level Navigation Decision Engine, routing high-level commands (`FORWARD`, `AVOID_OBSTACLE`) to `app/gait_controller.py`.

- **Phase 9: Raspberry Pi 5 Optimization**
  - Export trained models to ONNX / NCNN FP16 formats for fast edge execution ($\ge 20\text{ FPS}$) on Raspberry Pi 5.

- **Phase 10: Real-World Testing & Autonomous Field Deployment**
  - Conduct full autonomous indoor and outdoor navigation field trials with SpotMicro executing dynamic obstacle avoidance and stair/door navigation.

---

## 5. Detailed Component Plans

- Detailed Object Detection Plan: [dataset/detection/DATASET_PLAN.md](file:///c:/Users/Sreejith/spotmicro/dataset/detection/DATASET_PLAN.md)
- Detailed Navigation & Sensor Plan: [dataset/navigation/DATASET_PLAN.md](file:///c:/Users/Sreejith/spotmicro/dataset/navigation/DATASET_PLAN.md)
- Detection YOLO Configuration: [dataset/detection/data.yaml](file:///c:/Users/Sreejith/spotmicro/dataset/detection/data.yaml)
