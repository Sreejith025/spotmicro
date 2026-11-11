import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Base Paths
BASE_DIR = Path(r"D:\spotmicro")
NYU_DIR = BASE_DIR / "dataset" / "public" / "nyu_depth_v2"
EXTRACTED_RGB_DIR = NYU_DIR / "extracted" / "images"
EXTRACTED_LBL_DIR = NYU_DIR / "extracted" / "labels"
MAT_PATH = NYU_DIR / "nyu_depth_v2_labeled.mat"

OUTPUT_DIR = BASE_DIR / "dataset" / "detection_nyu_stage"
COCO_STAGE_DIR = BASE_DIR / "dataset" / "detection_coco_stage"
RUGD_STAGE_DIR = BASE_DIR / "dataset" / "detection_rugd_stage"

# Destination directories for NYU stage
TRAIN_IMG_DIR = OUTPUT_DIR / "images" / "train"
VAL_IMG_DIR = OUTPUT_DIR / "images" / "val"
TRAIN_LBL_DIR = OUTPUT_DIR / "labels" / "train"
VAL_LBL_DIR = OUTPUT_DIR / "labels" / "val"

# Class Mappings
# NYU ID 28 (door) -> YOLO Class 4
# NYU ID 215 (stairs) -> YOLO Class 3
NYU_CLASS_MAPPING = {
    28: 4,   # door -> YOLO class 4
    215: 3,  # stairs -> YOLO class 3
}

MIN_AREA_PIXELS = 100  # Minimum connected component area in pixels
SPLIT_RATIO = 0.8       # 80% train, 20% validation
RANDOM_SEED = 42

def convert_nyu_to_yolo():
    print("=" * 60)
    print("NYU Depth V2 to YOLO Bounding Box Converter for SpotMicro")
    print("=" * 60)

    # 1. Safety Verification
    print(f"COCO Staging Directory (Must NOT be modified) : {COCO_STAGE_DIR}")
    print(f"RUGD Staging Directory (Must NOT be modified) : {RUGD_STAGE_DIR}")
    print(f"NYU Output Staging Directory                 : {OUTPUT_DIR}")

    assert OUTPUT_DIR != COCO_STAGE_DIR, "Error: NYU output directory must be separate from COCO staging!"
    assert OUTPUT_DIR != RUGD_STAGE_DIR, "Error: NYU output directory must be separate from RUGD staging!"

    # Clean create NYU staging output directories
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Record initial timestamp/stat of original NYU dataset files to confirm untouched state
    mat_stat_before = MAT_PATH.stat() if MAT_PATH.exists() else None

    # Get sorted list of extracted RGB files
    rgb_files = sorted(list(EXTRACTED_RGB_DIR.glob("*.png")))
    total_images_processed = len(rgb_files)

    print(f"\nTotal extracted NYU images found: {total_images_processed}")

    # Deterministic train/val split using random seed 42
    random.seed(RANDOM_SEED)
    shuffled_files = list(rgb_files)
    random.shuffle(shuffled_files)

    num_train = int(total_images_processed * SPLIT_RATIO)
    train_set = set(p.name for p in shuffled_files[:num_train])
    val_set = set(p.name for p in shuffled_files[num_train:])

    print(f"Train images count      : {len(train_set)}")
    print(f"Validation images count : {len(val_set)}")
    print(f"Minimum component area  : {MIN_AREA_PIXELS} pixels")

    images_with_door = set()
    images_with_stairs = set()
    total_door_boxes = 0
    total_stairs_boxes = 0

    empty_label_files_count = 0

    print("\nProcessing images and extracting connected-component bounding boxes...")

    for rgb_path in rgb_files:
        filename = rgb_path.name  # e.g., '0001.png'
        stem = rgb_path.stem      # e.g., '0001'
        lbl_path = EXTRACTED_LBL_DIR / filename

        is_train = filename in train_set

        dst_img_dir = TRAIN_IMG_DIR if is_train else VAL_IMG_DIR
        dst_lbl_dir = TRAIN_LBL_DIR if is_train else VAL_LBL_DIR

        # Copy original RGB image untouched
        dst_img_path = dst_img_dir / filename
        shutil.copy2(rgb_path, dst_img_path)

        # Read uint16 label image
        lbl_img = Image.open(lbl_path)
        lbl_arr = np.array(lbl_img)  # shape (480, 640), uint16
        img_h, img_w = lbl_arr.shape  # 480, 640

        frame_boxes = []

        for nyu_id, yolo_cls in NYU_CLASS_MAPPING.items():
            mask = (lbl_arr == nyu_id).astype(np.uint8)
            if not np.any(mask):
                continue

            num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            for stat in stats[1:]:  # Skip background index 0
                x_min = stat[cv2.CC_STAT_LEFT]
                y_min = stat[cv2.CC_STAT_TOP]
                box_w = stat[cv2.CC_STAT_WIDTH]
                box_h = stat[cv2.CC_STAT_HEIGHT]
                area = stat[cv2.CC_STAT_AREA]

                if area < MIN_AREA_PIXELS:
                    continue

                if nyu_id == 28:
                    images_with_door.add(filename)
                    total_door_boxes += 1
                elif nyu_id == 215:
                    images_with_stairs.add(filename)
                    total_stairs_boxes += 1

                x_center = (x_min + box_w / 2.0) / img_w
                y_center = (y_min + box_h / 2.0) / img_h
                norm_w = box_w / img_w
                norm_h = box_h / img_h

                x_center = max(0.0, min(1.0, x_center))
                y_center = max(0.0, min(1.0, y_center))
                norm_w = max(0.0, min(1.0, norm_w))
                norm_h = max(0.0, min(1.0, norm_h))

                yolo_line = f"{yolo_cls} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"
                frame_boxes.append(yolo_line)

        # Write label file (empty if no door or stairs found)
        dst_lbl_path = dst_lbl_dir / f"{stem}.txt"
        with open(dst_lbl_path, "w", encoding="utf-8") as f_lbl:
            for line in frame_boxes:
                f_lbl.write(line + "\n")

        if not frame_boxes:
            empty_label_files_count += 1

    # Print Summary Report
    print("\n" + "=" * 60)
    print("NYU TO YOLO CONVERSION SUMMARY REPORT")
    print("=" * 60)
    print(f"1. Total NYU RGB images processed : {total_images_processed}")
    print(f"2. Number of train images         : {len(train_set)}")
    print(f"3. Number of validation images    : {len(val_set)}")
    print(f"4. Number of images with door     : {len(images_with_door)}")
    print(f"5. Number of images with stairs   : {len(images_with_stairs)}")
    print(f"6. Total door bounding boxes      : {total_door_boxes}")
    print(f"7. Total stairs bounding boxes    : {total_stairs_boxes}")
    print(f"8. Number of empty label files    : {empty_label_files_count}")
    print("=" * 60)

    # Comprehensive Validation Audit
    print("\n" + "=" * 60)
    print("COMPREHENSIVE VALIDATION AUDIT")
    print("=" * 60)

    train_imgs = set(p.stem for p in TRAIN_IMG_DIR.glob("*.png"))
    train_lbls = set(p.stem for p in TRAIN_LBL_DIR.glob("*.txt"))
    val_imgs = set(p.stem for p in VAL_IMG_DIR.glob("*.png"))
    val_lbls = set(p.stem for p in VAL_LBL_DIR.glob("*.txt"))

    missing_train_lbls = train_imgs - train_lbls
    missing_train_imgs = train_lbls - train_imgs
    missing_val_lbls = val_imgs - val_lbls
    missing_val_imgs = val_lbls - val_imgs

    missing_pairs = len(missing_train_lbls) + len(missing_train_imgs) + len(missing_val_lbls) + len(missing_val_imgs)

    all_lbl_paths = list(TRAIN_LBL_DIR.glob("*.txt")) + list(VAL_LBL_DIR.glob("*.txt"))
    invalid_label_lines = 0
    coords_outside_range = 0

    all_w = []
    all_h = []

    for lbl_path in all_lbl_paths:
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_label_lines += 1
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                if cls_id not in (3, 4):
                    invalid_label_lines += 1

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    coords_outside_range += 1

                all_w.append(w)
                all_h.append(h)
            except ValueError:
                invalid_label_lines += 1

    min_w = min(all_w) if all_w else 0.0
    max_w = max(all_w) if all_w else 0.0
    min_h = min(all_h) if all_h else 0.0
    max_h = max(all_h) if all_h else 0.0

    # Confirm original dataset untouched
    mat_stat_after = MAT_PATH.stat() if MAT_PATH.exists() else None
    untouched_confirmed = (mat_stat_before == mat_stat_after)

    print(f"9.  Missing image/label pairs                   : {missing_pairs}")
    print(f"10. Invalid YOLO label lines                    : {invalid_label_lines}")
    print(f"11. Bounding boxes with coords outside 0–1       : {coords_outside_range}")
    print(f"12. Min bounding-box dimensions (W x H)         : {min_w:.6f} x {min_h:.6f}")
    print(f"    Max bounding-box dimensions (W x H)         : {max_w:.6f} x {max_h:.6f}")
    print(f"13. Original NYU dataset untouched confirmed    : {untouched_confirmed}")

    if missing_pairs == 0 and invalid_label_lines == 0 and coords_outside_range == 0 and untouched_confirmed:
        print("\nSUCCESS: NYU conversion and validation audit PASSED 100%!")
    else:
        print("\nWARNING: Audit detected issues.")
    print("=" * 60)

if __name__ == "__main__":
    convert_nyu_to_yolo()
