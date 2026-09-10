# Object Detection Dataset Plan for SpotMicro

## 1. Executive Summary & Perception Strategy

For the **SpotMicro quadruped robot**, visual perception is the primary sensing modality for identifying dynamic entities (people, other robots) and environmental landmarks/hazards (chairs, stairs, doors, obstacles). 

The goal of this object detection plan is to establish a lightweight, highly efficient vision dataset strategy tailored for real-time edge inference on a **Raspberry Pi 5**. The detected bounding boxes and class probabilities feed directly into the high-level **Navigation Decision Engine**, which then dictates discrete movement actions (`FORWARD`, `BACKWARD`, `TURN_LEFT`, `TURN_RIGHT`, `STOP`, `AVOID_OBSTACLE`) to the low-level `GaitController`.

> [!IMPORTANT]
> **Edge Optimization Rule**: All detection models trained on this dataset must be optimized for execution on Raspberry Pi 5 using low-parameter architectures (e.g., YOLOv8n / YOLOv11n exported to ONNX or NCNN format) to achieve $\ge 15\text{--}30\text{ FPS}$ at $416\times 416$ or $640\times 640$ resolution.

---

## 2. Target Classes & Operational Definitions

The vision system focuses on 6 primary core classes and key secondary navigation classes:

| Class ID | Class Name | Category | Navigation Relevance for SpotMicro |
| :---: | :--- | :--- | :--- |
| `0` | **person** | Dynamic Obstacle | High priority collision avoidance; human following / interaction |
| `1` | **obstacle** | General Hazard | Boxes, trash cans, dynamic objects blocking ground movement |
| `2` | **chair** | Furniture / Path | Common indoor obstacle requiring gait steering around legs |
| `3` | **stairs** | Elevation Transition | Signals gait controller to switch to stair-climbing or step mode |
| `4` | **door** | Entryway / Waypoint | Passageway detection for indoor room-to-room navigation |
| `5` | **robot** | Multi-Agent | Identification of secondary SpotMicro units or ground robots |

---

## 3. Public Detection Datasets Evaluation

To avoid collecting tens of thousands of custom images from scratch, we leverage targeted subsets of established public datasets to train foundational feature representations before fine-tuning on college campus data.

### 3.1 COCO (Common Objects in Context)
- **Official URL**: [https://cocodataset.org/](https://cocodataset.org/)
- **Environment**: Both Indoor & Outdoor
- **Data Included**: 330k images, 80 object categories, 1.5 million object instances.
- **Useful Classes**: `person`, `chair`, `dining table`, `couch`, `potted plant`, `bottle`.
- **Approximate Storage**: ~25 GB full train/val set.
- **License**: CC BY 4.0
- **SpotMicro Relevance**: Excellent baseline weights for general person and furniture detection.
- **Priority**: **HIGH**
- **Download Strategy**: **SUBSET ONLY**. Extract only images containing `person` and `chair` via scripts or FiftyOne dataset tools (~1.5 GB). Do NOT download the full 25 GB dataset.

---

### 3.2 Open Images V7 (Google)
- **Official URL**: [https://storage.googleapis.com/openimages/web/index.html](https://storage.googleapis.com/openimages/web/index.html)
- **Environment**: Both Indoor & Outdoor
- **Data Included**: 9 million images annotated with image-level labels, object bounding boxes (600 categories).
- **Useful Classes**: `Door`, `Stairs`, `Chair`, `Person`, `Robot`, `Waste container`.
- **Approximate Storage**: >500 GB (Full).
- **License**: CC BY 2.0 / CC BY-SA 2.0
- **SpotMicro Relevance**: Valuable source for specialized classes like `Stairs` and `Door` which are under-represented in standard COCO.
- **Priority**: **MEDIUM**
- **Download Strategy**: **SUBSET ONLY**. Use official Python downloader tools to query and download only 1,000–2,000 images for `Stairs`, `Door`, and `Robot`. NEVER download full dataset.

---

### 3.3 Roboflow Universe Public Navigation Datasets
- **Official URL**: [https://universe.roboflow.com/](https://universe.roboflow.com/)
- **Environment**: Both Indoor & Outdoor
- **Data Included**: Pre-annotated community datasets specifically focused on robotics tasks (e.g., indoor door detection, stair step detection, indoor obstacle avoidance).
- **Useful Classes**: `door_handle`, `door_open`, `door_closed`, `stairs_step`, `robot_quadruped`, `low_obstacle`.
- **Approximate Storage**: 100 MB – 1 GB per curated dataset.
- **License**: Varies per dataset (CC BY 4.0 / Public Domain).
- **SpotMicro Relevance**: Highly relevant because annotations are pre-converted to YOLO format and focused on floor-level perspectives.
- **Priority**: **HIGH**
- **Download Strategy**: **SUBSET ONLY**. Download 2–3 specific highly-rated robot navigation projects (stair detection & door detection).

---

### 3.4 Pascal VOC 2012
- **Official URL**: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
- **Environment**: Both Indoor & Outdoor
- **Data Included**: 11,530 images annotated across 20 object classes.
- **Useful Classes**: `person`, `chair`, `sofa`, `diningtable`.
- **Approximate Storage**: ~2 GB.
- **License**: Open for non-commercial / research.
- **SpotMicro Relevance**: Low incremental value over COCO.
- **Priority**: **LOW**
- **Download Strategy**: **DO NOT DOWNLOAD**. COCO and Roboflow provide far superior coverage.

---

## 4. Class Extraction & Subsetting Strategy

To keep dataset storage lightweight ($\le 3\text{ GB}$ total on development machine), dataset subsetting is strictly enforced:

```text
[COCO Dataset]        ──► Extract 'person', 'chair' ────────┐
[Open Images V7]      ──► Extract 'stairs', 'door'  ────────┼──► [SpotMicro Perception Subset]
[Roboflow Navigation] ──► Extract 'obstacle', 'robot' ──────┘      (~3,000 images total)
```

### Scripted Subset Filtering Method
Using tools such as `FiftyOne` or custom Python scripts, dataset filtering rule:
$$\text{Include Image if } \exists c \in \{\text{person}, \text{obstacle}, \text{chair}, \text{stairs}, \text{door}, \text{robot}\}$$

---

## 5. Annotation Specifications & Bounding Box Standard

All custom dataset additions (from college campus recordings) must adhere strictly to standard **YOLO text format**:

```text
<class_id> <x_center> <y_center> <width> <height>
```

### Coordinate Normalization Equations
Given image pixel dimensions $W$ (width) and $H$ (height), and bounding box pixel coordinates $(x_{min}, y_{min}, x_{max}, y_{max})$:

$$\begin{aligned}
x_{center} &= \frac{x_{min} + x_{max}}{2 \cdot W} \\
y_{center} &= \frac{y_{min} + y_{max}}{2 \cdot H} \\
width &= \frac{x_{max} - x_{min}}{W} \\
height &= \frac{y_{max} - y_{min}}{H}
\end{aligned}$$

All values MUST satisfy $0.0 \le x_{center}, y_{center}, width, height \le 1.0$.

---

## 6. Real-World Image Collection Guidelines for SpotMicro

To make the object detector resilient when mounted on the actual robot:

1. **Low Camera Mount Angle**: Cameras must be mounted at $15\text{--}25\text{ cm}$ height from ground level to match actual SpotMicro payload height.
2. **Camera Pitch Variance**: Collect images with horizontal, upward pitch ($+15^\circ$), and downward pitch ($-20^\circ$) to simulate robot body pitch during walk/trot gaits.
3. **Lighting Diversity**:
   - Indoor: Fluorescent lighting, dim hallway lighting, sunlight cast through windows.
   - Outdoor: Direct sunlight, shade under trees, overcast skies, sunset/dusk.
4. **Distance Profiles**:
   - Near field ($0.3\text{ m} - 1.0\text{ m}$): Urgent obstacle avoidance zone.
   - Mid field ($1.0\text{ m} - 3.0\text{ m}$): Path planning and steering zone.
   - Far field ($3.0\text{ m} - 6.0\text{ m}$): Long-range waypoint targeting.

---

## 7. Edge Deployment Strategy (Raspberry Pi 5)

```text
[Trained PyTorch YOLO Model] 
            ↓ (Export)
   [ONNX / NCNN Format] 
            ↓ (Quantization)
   [FP16 / INT8 Precision] 
            ↓ (Deploy)
[Raspberry Pi 5 Execution] ──► Output Bounding Boxes ──► [Navigation Engine]
```

- **Target Model Architecture**: YOLOv8n / YOLOv11n (Nano variant, ~3M parameters).
- **Target Resolution**: $416 \times 416$ or $640 \times 640$.
- **Expected Raspberry Pi 5 Performance**: $20\text{--}35\text{ FPS}$ using OpenCV dnn / ONNX Runtime with ARM NEON / V3D acceleration.
