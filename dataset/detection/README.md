# SpotMicro Autonomous Navigation - Object Detection Dataset

## 1. Purpose
The primary purpose of this dataset is to enable custom object detection capabilities for the **SpotMicro quadruped robot**. By recognizing surroundings and key objects, SpotMicro can navigate autonomously, avoid obstacles, locate passage pathways (such as doors and stairs), and operate safely around people and other robots.

## 2. Object Classes
The dataset specifies **6 object classes** indexed from 0 to 5:

| Class ID | Class Name | Description |
| :--- | :--- | :--- |
| `0` | **person** | Humans in various postures (standing, sitting, walking) |
| `1` | **obstacle** | Objects obstructing navigation (boxes, trash cans, barriers) |
| `2` | **chair** | Seating furniture (chairs, stools, desk chairs) |
| `3` | **stairs** | Staircases and steps for elevation awareness and gait adjustment |
| `4` | **door** | Doors, doorways, and open/closed entrances |
| `5` | **robot** | SpotMicro units or other robotic platforms |

## 3. YOLO Bounding-Box Annotation Format
Annotations follow the standard **YOLO text format** with one line per object bounding box:

```text
class_id x_center y_center width height
```

Example annotation entry:
```text
0 0.485000 0.342000 0.180000 0.450000
```

## 4. Coordinate Normalization
- All bounding box spatial parameters (`x_center`, `y_center`, `width`, `height`) are **normalized to values between 0.0 and 1.0** relative to the total width and height of the image.
- $\text{x\_center} = \frac{\text{box center X}}{\text{image width}}$
- $\text{y\_center} = \frac{\text{box center Y}}{\text{image height}}$
- $\text{width} = \frac{\text{box width}}{\text{image width}}$
- $\text{height} = \frac{\text{box height}}{\text{image height}}$

## 5. Dataset Split Ratio
The recommended dataset distribution is:
- **70% Training (`images/train`, `labels/train`)**: Used for learning feature representations and model weights.
- **20% Validation (`images/val`, `labels/val`)**: Used for hyperparameter tuning and preventing overfitting during training.
- **10% Testing (`images/test`, `labels/test`)**: Reserved for final evaluation of model accuracy and generalization.

## 6. Image Collection Environment
- All dataset images will be captured directly from our **college environment**, including hallways, classrooms, laboratories, stairwells, entrances, and campus walkways.

## 7. Dataset Diversity & Quality Guidelines
To ensure robust real-world inference performance, images must capture diverse conditions:
- **Distances**: Close-range, medium-range, and long-range perspectives.
- **Angles**: High, low, side, and robot-perspective viewpoints (low camera height matching SpotMicro's camera placement).
- **Lighting Conditions**: Daylight, artificial indoor lighting, low light, shadows, and high dynamic range scenes.
- **Backgrounds**: Complex indoor backgrounds, plain walls, outdoor textures, dynamic human movement.
- **Object Positions**: Centered, off-center, partial occlusions, and overlapping objects.

## 8. Quality Assurance & Manual Inspection
- All dataset annotations will be **manually reviewed and verified** before initiating model training to ensure bounding box accuracy and eliminate mislabeling.

## 9. Model Training Strategy
- AI model training will be conducted on a **laptop or cloud GPU server** initially. Training will not run on the Raspberry Pi during dataset preparation or initial model iteration.

## 10. Deployment Target
- The final trained and optimized model will be deployed to the **Raspberry Pi 5** on the SpotMicro quadruped for real-time edge object detection and dynamic autonomous navigation.
