import os
import random
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

# Base Paths
BASE_DIR = Path(r"D:\spotmicro")
RGB_DIR = BASE_DIR / "dataset" / "public" / "rugd" / "RUGD_frames-with-annotations"
MASK_DIR = BASE_DIR / "dataset" / "public" / "rugd" / "RUGD_annotations"
if (MASK_DIR / "RUGD_annotations").exists():
    MASK_DIR = MASK_DIR / "RUGD_annotations"

OUTPUT_DIR = BASE_DIR / "dataset" / "detection_rugd_stage"
COCO_STAGE_DIR = BASE_DIR / "dataset" / "detection_coco_stage"
NYU_STAGE_DIR = BASE_DIR / "dataset" / "detection_nyu_stage"

# Destination directories for RUGD stage
TRAIN_IMG_DIR = OUTPUT_DIR / "images" / "train"
VAL_IMG_DIR = OUTPUT_DIR / "images" / "val"
TRAIN_LBL_DIR = OUTPUT_DIR / "labels" / "train"
VAL_LBL_DIR = OUTPUT_DIR / "labels" / "val"

# Target RUGD Obstacle Classes & OpenCV BGR Palette (B, G, R)
# ALL target obstacle classes map to YOLO Class 1 (obstacle)
TARGET_OBSTACLE_CLASSES = {
    "rock":      (160, 160, 160),
    "pole":      (0, 153, 153),
    "container": (255, 0, 255),
    "debris":    (0, 255, 255),
    "object":    (0, 128, 255),
    "log":       (0, 51, 102),
    "fence":     (143, 143, 188),
}

YOLO_OBSTACLE_CLASS_ID = 1
MIN_AREA_PIXELS = 100  # Minimum connected component area in pixels
RANDOM_SEED = 42
SPLIT_RATIO = 0.8  # ~80% train sequences, ~20% val sequences

def process_single_frame(args):
    seq_name, rgb_path, is_train = args
    mask_path = MASK_DIR / seq_name / rgb_path.name

    if not mask_path.exists():
        return None

    # Load mask image in BGR mode
    mask_bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if mask_bgr is None:
        return None

    img_h, img_w, _ = mask_bgr.shape  # 550, 688
    frame_boxes = []
    class_counts = {k: 0 for k in TARGET_OBSTACLE_CLASSES}

    for cls_name, (b, g, r) in TARGET_OBSTACLE_CLASSES.items():
        # Fast binary mask matching in BGR
        binary_mask = (mask_bgr[:, :, 0] == b) & (mask_bgr[:, :, 1] == g) & (mask_bgr[:, :, 2] == r)

        if not np.any(binary_mask):
            continue

        binary_mask_u8 = binary_mask.astype(np.uint8)

        # Connected components analysis
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(binary_mask_u8, connectivity=8)

        for stat in stats[1:]:  # Skip background index 0
            x_min = stat[cv2.CC_STAT_LEFT]
            y_min = stat[cv2.CC_STAT_TOP]
            box_w = stat[cv2.CC_STAT_WIDTH]
            box_h = stat[cv2.CC_STAT_HEIGHT]
            area = stat[cv2.CC_STAT_AREA]

            if area < MIN_AREA_PIXELS:
                continue

            class_counts[cls_name] += 1

            # Normalize bounding box coordinates to YOLO format [0, 1]
            x_center = (x_min + box_w / 2.0) / img_w
            y_center = (y_min + box_h / 2.0) / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h

            # Clamp coordinates to [0.0, 1.0]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            norm_w = max(0.0, min(1.0, norm_w))
            norm_h = max(0.0, min(1.0, norm_h))

            yolo_line = f"{YOLO_OBSTACLE_CLASS_ID} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
            frame_boxes.append(yolo_line)

    if not frame_boxes:
        return (seq_name, is_train, 0, 0, class_counts)

    # Destination file paths
    dst_img_dir = TRAIN_IMG_DIR if is_train else VAL_IMG_DIR
    dst_lbl_dir = TRAIN_LBL_DIR if is_train else VAL_LBL_DIR

    # Unique filename containing sequence name
    unique_name = f"{seq_name}_{rgb_path.name}"
    lbl_name = Path(unique_name).stem + ".txt"

    dst_img_path = dst_img_dir / unique_name
    dst_lbl_path = dst_lbl_dir / lbl_name

    # Copy RGB frame image
    shutil.copy2(rgb_path, dst_img_path)

    # Write YOLO label file
    with open(dst_lbl_path, "w", encoding="utf-8") as f_lbl:
        for line in frame_boxes:
            f_lbl.write(line + "\n")

    return (seq_name, is_train, 1, len(frame_boxes), class_counts)

def convert_rugd_to_yolo():
    print("=" * 60)
    print("RUGD Obstacle to YOLO Bounding Box Converter for SpotMicro")
    print("=" * 60)

    # Safety verification
    print(f"COCO Staging Directory (Must NOT be modified) : {COCO_STAGE_DIR}")
    print(f"NYU Staging Directory (Must NOT be modified)  : {NYU_STAGE_DIR}")
    print(f"RUGD Output Staging Directory                 : {OUTPUT_DIR}")

    assert OUTPUT_DIR != COCO_STAGE_DIR, "Error: RUGD output directory must be separate from COCO staging!"
    assert OUTPUT_DIR != NYU_STAGE_DIR, "Error: RUGD output directory must be separate from NYU staging!"

    # Clean create RUGD staging directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Get sequence directories and deterministic sequence split
    seq_dirs = sorted([d for d in RGB_DIR.iterdir() if d.is_dir()])
    seq_names = [d.name for d in seq_dirs]

    random.seed(RANDOM_SEED)
    shuffled_seqs = list(seq_names)
    random.shuffle(shuffled_seqs)

    num_train_seqs = int(len(shuffled_seqs) * SPLIT_RATIO)
    train_seq_set = set(shuffled_seqs[:num_train_seqs])
    val_seq_set = set(shuffled_seqs[num_train_seqs:])

    print(f"\nTotal Sequence Folders Found: {len(seq_names)}")
    print(f"Train Sequences ({len(train_seq_set)}): {sorted(list(train_seq_set))}")
    print(f"Val Sequences   ({len(val_seq_set)}): {sorted(list(val_seq_set))}")

    tasks = []
    for seq_name in sorted(seq_names):
        seq_rgb_dir = RGB_DIR / seq_name
        if not seq_rgb_dir.exists():
            continue
        is_train = seq_name in train_seq_set
        for rgb_path in sorted(list(seq_rgb_dir.glob("*.png"))):
            tasks.append((seq_name, rgb_path, is_train))

    total_frames_inspected = len(tasks)
    print(f"\nProcessing {total_frames_inspected} annotation masks with multithreading (MIN_AREA = {MIN_AREA_PIXELS} pixels)...")

    boxes_by_class = {k: 0 for k in TARGET_OBSTACLE_CLASSES}
    total_obstacle_frames = 0
    total_obstacle_boxes = 0
    train_img_count = 0
    val_img_count = 0
    train_box_count = 0
    val_box_count = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(process_single_frame, tasks)

    for res in results:
        if res is None:
            continue
        seq_name, is_train, is_obs_frame, num_boxes, class_counts = res

        for k, v in class_counts.items():
            boxes_by_class[k] += v

        if is_obs_frame:
            total_obstacle_frames += 1
            total_obstacle_boxes += num_boxes

            if is_train:
                train_img_count += 1
                train_box_count += num_boxes
            else:
                val_img_count += 1
                val_box_count += num_boxes

    # Print Summary
    print("\n" + "=" * 60)
    print("RUGD TO YOLO OBSTACLE CONVERSION SUMMARY")
    print("=" * 60)
    print(f"total RUGD frames inspected       : {total_frames_inspected}")
    print(f"total frames containing obstacles : {total_obstacle_frames}")
    print(f"total obstacle bounding boxes     : {total_obstacle_boxes}")
    print(f"minimum component area            : {MIN_AREA_PIXELS} pixels")
    print("\nBounding boxes by original RUGD class:")
    for cls_name, count in boxes_by_class.items():
        print(f"  {cls_name:<12}: {count:>6} boxes")

    print("\nSplit statistics:")
    print(f"  train image count               : {train_img_count}")
    print(f"  validation image count          : {val_img_count}")
    print(f"  train obstacle-box count        : {train_box_count}")
    print(f"  validation obstacle-box count   : {val_box_count}")

    print("\nSequence splits:")
    print(f"  train sequence names ({len(train_seq_set)}): {sorted(list(train_seq_set))}")
    print(f"  validation sequence names ({len(val_seq_set)}): {sorted(list(val_seq_set))}")
    print("=" * 60)

    # Post-Conversion Validation Audit
    print("\n" + "=" * 60)
    print("AUTOMATIC POST-CONVERSION VALIDATION AUDIT")
    print("=" * 60)

    train_imgs = set(p.stem for p in TRAIN_IMG_DIR.glob("*.png"))
    train_lbls = set(p.stem for p in TRAIN_LBL_DIR.glob("*.txt"))
    val_imgs = set(p.stem for p in VAL_IMG_DIR.glob("*.png"))
    val_lbls = set(p.stem for p in VAL_LBL_DIR.glob("*.txt"))

    missing_train_lbls = train_imgs - train_lbls
    missing_train_imgs = train_lbls - train_imgs
    missing_val_lbls = val_imgs - val_lbls
    missing_val_imgs = val_lbls - val_imgs

    missing_pairs_count = len(missing_train_lbls) + len(missing_train_imgs) + len(missing_val_lbls) + len(missing_val_imgs)

    all_lbl_paths = list(TRAIN_LBL_DIR.glob("*.txt")) + list(VAL_LBL_DIR.glob("*.txt"))
    empty_labels_count = 0
    invalid_labels_count = 0
    non_class1_count = 0

    for lbl_path in all_lbl_paths:
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            empty_labels_count += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_labels_count += 1
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
                if cls_id != YOLO_OBSTACLE_CLASS_ID:
                    non_class1_count += 1
                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    invalid_labels_count += 1
            except ValueError:
                invalid_labels_count += 1

    print(f"Missing image/label pairs  : {missing_pairs_count}")
    print(f"Empty label files           : {empty_labels_count}")
    print(f"Invalid label lines         : {invalid_labels_count}")
    print(f"Non-Class-1 annotations     : {non_class1_count}")

    if missing_pairs_count == 0 and empty_labels_count == 0 and invalid_labels_count == 0 and non_class1_count == 0:
        print("\nSUCCESS: Dataset conversion and post-validation audit PASSED 100%!")
    else:
        print("\nWARNING: Audit detected issues that need review.")
    print("=" * 60)

if __name__ == "__main__":
    convert_rugd_to_yolo()
