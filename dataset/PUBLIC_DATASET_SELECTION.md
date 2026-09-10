# SpotMicro Autonomous Navigation - Public Dataset Selection Report

This document presents the concrete dataset selection and data-minimization strategy for the **SpotMicro quadruped robot** autonomous navigation system.

---

## 1. System Requirements & Perception-Control Breakdown

To achieve autonomous indoor and outdoor navigation, SpotMicro relies on the following sensor-hardware setup:
- **RGB Camera**: Monocular front-facing camera for perception and depth estimation.
- **BNO055 IMU**: 9-DOF sensor providing Euler angles, quaternions, angular velocity, and linear acceleration for stability.
- **HC-SR04 Ultrasonic Sensors**: Front distance measurement for immediate near-field collision fallback.
- **Raspberry Pi 5**: Onboard edge processor executing quantized inference models.
- **PCA9685 + 12 x DS3225 Servos**: Low-level motor hardware driven by `GaitController` (IK and gait generation).

### Functional Category Breakdown & Data Requirements

| Category | Function | AI / Control Module | Primary Dataset Requirement |
| :--- | :--- | :--- | :--- |
| **A. Object Detection** | Identifies `person`, `obstacle`, `chair`, `stairs`, `door`, `robot`. | YOLOv8n / YOLOv11n | 2D bounding boxes in YOLO format (`class_id x y w h`) |
| **B. Depth / Perception** | Estimates per-pixel distance from single RGB camera. | MiDaS / MobileNet-Depth | Aligned RGB + Dense Depth pairs |
| **C. Indoor Navigation** | Navigates hallways, doorways, and rooms. | Navigation State Machine | Indoor topology, door/stair landmarks, RGB-D paths |
| **D. Outdoor Navigation** | Navigates outdoor paths, avoiding off-road hazards. | Navigation Decision Engine | Ground-level outdoor paths, dynamic obstacle sequences |
| **E. Terrain Understanding** | Classifies surface type (`grass`, `gravel`, `dirt`, `asphalt`). | Surface Classifier / Segmenter | Pixel-level semantic terrain masks & traversability maps |
| **F. Robot Motion / State** | Monitors body pitch/roll, quadruped pose, gait stability. | State Estimator & Stability Guard | Synchronized IMU, pose trajectories, quadruped dynamics |
| **G. Real-Robot Telemetry** | Final fine-tuning & real-world operational safety. | SpotMicro Logging Framework | Real camera frames, BNO055 IMU, Ultrasonic, Servo feedback |

---

## 2. Comprehensive Evaluation of Candidate Datasets

### 2.1 TartanGround
- **Official Source URL**: [https://theairlab.org/tartanground-dataset/](https://theairlab.org/tartanground-dataset/)
- **Domain**: Indoor & Outdoor Ground Robot Trajectories
- **Data Available**: RGB (Monocular & Stereo), Depth, IMU, Ground Truth Pose, Semantic Segmentation.
- **LiDAR / Bounding Boxes**: No LiDAR required; No native bounding boxes.
- **Terrain Information**: Detailed ground traversability, surface transitions, obstacle boundaries.
- **Quadruped / Ground Relevance**: **Extremely High** — Cameras were mounted at ground-level height ($15\text{--}30\text{ cm}$), perfectly matching SpotMicro's camera height.
- **Storage Size**: Full: ~150 GB | **Subset Recommended: ~8.0 GB**
- **License**: CC BY 4.0 (Research Open Access)
- **Selection**: **SUBSET ONLY** (Download 2-3 ground navigation sequences).
- **Priority**: **HIGH**

---

### 2.2 TartanAir
- **Official Source URL**: [https://theairlab.org/tartanair-dataset/](https://theairlab.org/tartanair-dataset/)
- **Domain**: Photorealistic Synthetic Indoor & Outdoor Environments (Unreal Engine)
- **Data Available**: Stereo RGB, Exact Depth, Disparity, Optical Flow, Camera Trajectories, Simulated IMU.
- **LiDAR / Bounding Boxes**: No LiDAR; No bounding boxes.
- **Terrain Information**: Synthetic surfaces, slopes, structural obstacles.
- **Quadruped / Ground Relevance**: **High for depth & SLAM pre-training** — Offers perfectly clean depth maps across challenging lighting/weather (fog, rain, night).
- **Storage Size**: Full: ~3 TB | **Subset Recommended: ~5.0 GB**
- **License**: CC BY 4.0
- **Selection**: **SUBSET ONLY** (1 indoor `office` sequence and 1 outdoor `neighborhood` sequence).
- **Priority**: **MEDIUM**

---

### 2.3 HM3D / HM3DSem (Habitat-Matterport 3D)
- **Official Source URL**: [https://aihabitat.org/datasets/hm3d/](https://aihabitat.org/datasets/hm3d/)
- **Domain**: Indoor (1,000 real-world 3D scanned residential & commercial buildings)
- **Data Available**: Renderable RGB-D via Habitat-Sim, 3D Semantic Annotations (`door`, `stairs`, `chair`, `table`, `wall`).
- **LiDAR / IMU**: No LiDAR; No hardware IMU.
- **Terrain Information**: Indoor floor surfaces, steps, doorways, indoor obstacles.
- **Quadruped / Ground Relevance**: **High for Indoor Passage Traversal** — Provides realistic indoor navigation topologies (hallways, doorways, staircases).
- **Storage Size**: Raw 3D Meshes: ~200 GB | **Rendered Subset Recommended: ~4.0 GB**
- **License**: FAIR Habitat License (Academic / Research)
- **Selection**: **SUBSET ONLY** (Render 2D RGB-D images along robot paths; DO NOT download raw 200 GB mesh assets).
- **Priority**: **MEDIUM**

---

### 2.4 NYU Depth V2
- **Official Source URL**: [https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- **Domain**: Indoor (464 diverse indoor scenes)
- **Data Available**: 1,449 aligned RGB + Kinect Depth pairs, 40-class semantic segmentation (`chair`, `door`, `stairs`, `table`, `wall`, `floor`).
- **LiDAR / IMU**: No LiDAR; Raw accelerometer available.
- **Terrain Information**: Indoor flooring types, carpets, stairs, steps.
- **Quadruped / Ground Relevance**: **Very High** — Standard lightweight benchmark for indoor monocular depth estimation.
- **Storage Size**: **Full Dataset: 2.8 GB**
- **License**: Open Source (BSD)
- **Selection**: **COMPLETE DATASET** (Compact size makes full download practical).
- **Priority**: **HIGH**

---

### 2.5 COCO (Common Objects in Context 2017)
- **Official Source URL**: [https://cocodataset.org/](https://cocodataset.org/)
- **Domain**: Indoor & Outdoor General Object Detection
- **Data Available**: 330k RGB images, 80 object categories with 2D bounding boxes and polygon masks.
- **Depth / IMU / Pose**: None.
- **Terrain Information**: None.
- **Quadruped / Ground Relevance**: **High for Object Detection Baseline** — Essential baseline weights for detecting `person` and `chair`.
- **Storage Size**: Full: ~25 GB | **Class Filtered Subset Recommended: ~1.5 GB**
- **License**: CC BY 4.0
- **Selection**: **SUBSET ONLY** (Filter and download only images containing `person` and `chair`).
- **Priority**: **HIGH**

---

### 2.6 Open Images V7 (Google)
- **Official Source URL**: [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
- **Domain**: Indoor & Outdoor
- **Data Available**: 9 million RGB images, 600 box categories (`Door`, `Stairs`, `Chair`, `Person`, `Robot`).
- **Depth / IMU / Pose**: None.
- **Terrain Information**: Minimal.
- **Quadruped / Ground Relevance**: **Medium for Target Class Augmentation** — Fills gaps for underrepresented object categories like `Stairs`, `Door`, and `Robot`.
- **Storage Size**: Full: >500 GB | **Targeted Subset Recommended: ~2.0 GB**
- **License**: CC BY 2.0 / CC BY-SA 2.0
- **Selection**: **SUBSET ONLY** (Query and download only ~1,000 images for `Stairs`, `Door`, and `Robot`).
- **Priority**: **MEDIUM**

---

### 2.7 RUGD (Robot Unstructured Ground Dataset)
- **Official Source URL**: [https://rugd.cais.wisc.edu/](https://rugd.cais.wisc.edu/)
- **Domain**: Outdoor Off-Road Paths, Parks, Campus Trails
- **Data Available**: 7,500+ sequence frames captured from ground robot viewpoint with fine semantic segmentation (`grass`, `gravel`, `dirt`, `asphalt`, `bush`, `rock`, `obstacle`).
- **Depth / IMU**: Monocular RGB; Video sequence continuity.
- **Terrain Information**: **Extensive ground traversability data**.
- **Quadruped / Ground Relevance**: **Essential for Outdoor Navigation** — Ground-level perspective matching SpotMicro outdoor path planning.
- **Storage Size**: **Full Dataset / Main Split: ~15.0 GB**
- **License**: Academic Research
- **Selection**: **COMPLETE DATASET / MAIN SPLIT**
- **Priority**: **HIGH**

---

### 2.8 RELLIS-3D (Unmanned Systems Lab)
- **Official Source URL**: [https://unmanned-lab.github.io/rellis-3d](https://unmanned-lab.github.io/rellis-3d)
- **Domain**: Outdoor Off-Road & Rough Terrain
- **Data Available**: High-resolution RGB video sequences, 6-DOF IMU, GPS, semantic segmentation for `mud`, `grass`, `rubble`, `obstacle`, `puddle`.
- **LiDAR**: 3D LiDAR point clouds available (*Excluded to save space*).
- **Terrain Information**: Rough terrain traversability and slope stability.
- **Quadruped / Ground Relevance**: **High for Outdoor Rough Terrain Navigation**.
- **Storage Size**: Full: ~110 GB | **RGB + IMU Subset Recommended: ~12.0 GB**
- **License**: Open Academic
- **Selection**: **SUBSET ONLY** (Download RGB camera frames + IMU logs + segmentation labels; discard heavy 3D LiDAR point clouds).
- **Priority**: **HIGH**

---

### 2.9 Roboflow Public Navigation Datasets
- **Official Source URL**: [https://universe.roboflow.com/](https://universe.roboflow.com/)
- **Domain**: Indoor & Outdoor Robotics Vision Projects
- **Data Available**: Community pre-labeled object detection datasets formatted directly in **YOLO text format** (`stairs`, `door`, `obstacle`, `robot`).
- **Depth / IMU / Pose**: None.
- **Terrain Information**: Focused on doors, stair steps, dynamic obstacles.
- **Quadruped / Ground Relevance**: **High for Custom Detection Targets** — Immediate integration into YOLO training.
- **Storage Size**: **Selected Projects: ~1.0 GB**
- **License**: CC BY 4.0 / Public Domain
- **Selection**: **SUBSET ONLY** (2-3 targeted projects).
- **Priority**: **HIGH**

---

### 2.10 ETH Zurich ANYmal Quadruped Datasets
- **Official Source URL**: [https://rsl.ethz.ch/](https://rsl.ethz.ch/) & [https://ori.ox.ac.uk/](https://ori.ox.ac.uk/)
- **Domain**: Quadruped Indoor & Outdoor Rough Terrain Navigation
- **Data Available**: Quadruped sensor logs: RGB-D camera, High-rate IMU (100 Hz+), Joint encoders, Foot contact sensors, Trajectories over stairs & slopes.
- **LiDAR**: Available (*Excluded*).
- **Terrain Information**: Quadruped gait dynamics, foot contact stability, body pitch/roll response.
- **Quadruped / Ground Relevance**: **Extremely High for Quadruped Motion & State Estimation** — Real quadruped telemetry reference.
- **Storage Size**: Full: ~35 GB | **Subset Recommended: ~5.0 GB**
- **License**: Academic Research
- **Selection**: **SUBSET ONLY** (3 sequences focusing on stair climbing & rough terrain stability).
- **Priority**: **HIGH**

---

## 3. Master Dataset Evaluation Table

| DATASET | PURPOSE | INDOOR/OUTDOOR | SUBSET TO DOWNLOAD | PRIORITY |
| :--- | :--- | :---: | :--- | :---: |
| **COCO** | Object Detection (`person`, `chair`) | Both | Filtered class subset (~1.5 GB) | **HIGH** |
| **Roboflow Nav** | Object Detection (`stairs`, `door`, `obstacle`, `robot`) | Both | Pre-annotated YOLO projects (~1.0 GB) | **HIGH** |
| **NYU Depth V2** | Monocular Depth Estimation | Indoor | Full dataset (~2.8 GB) | **HIGH** |
| **RUGD** | Path Traversability & Surface Segmentation | Outdoor | Full dataset / Main split (~15.0 GB) | **HIGH** |
| **RELLIS-3D** | Off-Road Rough Terrain & Obstacles | Outdoor | RGB + IMU + Labels subset (~12.0 GB) | **HIGH** |
| **TartanGround** | Ground-Level Visual Trajectories & Depth | Both | 2-3 ground sequences (~8.0 GB) | **HIGH** |
| **ETH ANYmal** | Quadruped IMU & Gait Dynamics | Both | 3 stair/rough terrain sequences (~5.0 GB) | **HIGH** |
| **Open Images V7**| Class Augmentation (`Stairs`, `Door`, `Robot`) | Both | Query target classes (~2.0 GB) | **MEDIUM** |
| **TartanAir** | Synthetic Depth & Visual SLAM | Both | 2 synthetic sequences (~5.0 GB) | **MEDIUM** |
| **HM3D / HM3DSem**| Indoor Room Traversal & Topology | Indoor | Rendered RGB-D frames (~4.0 GB) | **MEDIUM** |

---

## 4. Final Recommended Minimal Dataset Selection (The Core 5)

To achieve maximum coverage across perception, depth, terrain, and quadruped dynamics **without downloading terabytes of unnecessary data**, we select the following **5 core datasets**:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   THE CORE 5 PUBLIC DATASET SELECTION                   │
├──────────────────────┬─────────────────────────────┬───────────────────┤
│ Dataset              │ Focus Area                  │ Download Size     │
├──────────────────────┼─────────────────────────────┼───────────────────┤
│ 1. COCO + Roboflow   │ Object Detection (6 Classes)│ ~2.5 GB (Subset)  │
│ 2. NYU Depth V2      │ Indoor Depth Estimation     │ ~2.8 GB (Full)    │
│ 3. RUGD              │ Outdoor Surface & Terrain   │ ~15.0 GB (Full)   │
│ 4. RELLIS-3D         │ Outdoor Off-Road & IMU      │ ~12.0 GB (Subset) │
│ 5. TartanGround      │ Ground-Perspective Visuals  │ ~8.0 GB (Subset)  │
├──────────────────────┴─────────────────────────────┼───────────────────┤
│ TOTAL ESTIMATED DOWNLOAD STORAGE                   │ ~40.3 GB          │
└────────────────────────────────────────────────────┴───────────────────┘
```

---

## 5. What We Collect Ourselves (SpotMicro Telemetry)

Public datasets provide pre-training; our custom collection provides real deployment reliability:

1. **Low-Angle Camera Frames**: Captured at $15\text{--}25\text{ cm}$ height in college corridors, stairwells, and outdoor paths.
2. **BNO055 IMU Data**: Synchronized pitch, roll, yaw, linear acceleration, and angular velocity at $10\text{ Hz}$.
3. **HC-SR04 Ultrasonic Distance**: Front distance readings ($2\text{ cm} - 400\text{ cm}$) aligned with camera frames.
4. **Gait Controller States**: Commanded direction (`forward`, `left`, `stop`), stride frequency, and 12-servo joint angle feedback.
5. **Real Movement Outcomes**: Logged success/failure during obstacle avoidance maneuvers.

---

## 6. Strict Blacklist: What Must NOT Be Downloaded ❌

1. ❌ **Full COCO Dataset (25 GB)** — Filter classes to save 23.5 GB.
2. ❌ **Full Open Images V7 (>500 GB)** — Do NOT download full dump; use target queries only.
3. ❌ **Full TartanAir (3 TB)** — Strictly limit to 1-2 small sequences.
4. ❌ **Raw HM3D 3D Mesh Files (200 GB)** — Render 2D frames only via Habitat-Sim.
5. ❌ **Heavy 3D LiDAR Point Cloud Binaries from RELLIS-3D / Waymo (>100 GB)** — Exclude LiDAR files.
