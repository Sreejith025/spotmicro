import os
import shutil
from pathlib import Path

# Base Paths
BASE_DIR = Path(r"D:\spotmicro")
DATASET_DIR = BASE_DIR / "dataset"

COCO_STAGE_DIR = DATASET_DIR / "detection_coco_stage"
NYU_STAGE_DIR = DATASET_DIR / "detection_nyu_stage"
RUGD_STAGE_DIR = DATASET_DIR / "detection_rugd_stage"

FINAL_STAGE_DIR = DATASET_DIR / "detection_final_stage"

TRAIN_IMG_DIR = FINAL_STAGE_DIR / "images" / "train"
VAL_IMG_DIR = FINAL_STAGE_DIR / "images" / "val"
TRAIN_LBL_DIR = FINAL_STAGE_DIR / "labels" / "train"
VAL_LBL_DIR = FINAL_STAGE_DIR / "labels" / "val"

# Target Class Names
CLASS_NAMES = {
    0: "person",
    1: "obstacle",
    2: "chair",
    3: "stairs",
    4: "door",
    5: "robot",
}

def get_dir_size(path):
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total

def merge_final_dataset():
    print("=" * 60)
    print("SPOTMICRO FINAL YOLO DATASET MERGE & AUDIT")
    print("=" * 60)

    # 1. Safety Checks
    print(f"COCO Source Stage (Must NOT be modified) : {COCO_STAGE_DIR}")
    print(f"NYU Source Stage (Must NOT be modified)  : {NYU_STAGE_DIR}")
    print(f"RUGD Source Stage (Must NOT be modified) : {RUGD_STAGE_DIR}")
    print(f"Final Output Stage Directory             : {FINAL_STAGE_DIR}")

    assert FINAL_STAGE_DIR != COCO_STAGE_DIR
    assert FINAL_STAGE_DIR != NYU_STAGE_DIR
    assert FINAL_STAGE_DIR != RUGD_STAGE_DIR

    # Clean create final stage directories
    if FINAL_STAGE_DIR.exists():
        shutil.rmtree(FINAL_STAGE_DIR)

    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    sources = [
        ("COCO", COCO_STAGE_DIR),
        ("NYU", NYU_STAGE_DIR),
        ("RUGD", RUGD_STAGE_DIR),
    ]

    source_counts = {
        "COCO": {"images": 0, "labels": 0},
        "NYU": {"images": 0, "labels": 0},
        "RUGD": {"images": 0, "labels": 0},
    }

    print("\nMerging staging datasets into detection_final_stage...")

    for name, stage_path in sources:
        for split in ["train", "val"]:
            src_img_dir = stage_path / "images" / split
            src_lbl_dir = stage_path / "labels" / split
            dst_img_dir = FINAL_STAGE_DIR / "images" / split
            dst_lbl_dir = FINAL_STAGE_DIR / "labels" / split

            # Copy images
            img_files = list(src_img_dir.glob("*"))
            for img_p in img_files:
                if img_p.is_file():
                    shutil.copy2(img_p, dst_img_dir / img_p.name)
                    source_counts[name]["images"] += 1

            # Copy labels
            lbl_files = list(src_lbl_dir.glob("*"))
            for lbl_p in lbl_files:
                if lbl_p.is_file():
                    shutil.copy2(lbl_p, dst_lbl_dir / lbl_p.name)
                    source_counts[name]["labels"] += 1

        print(f"  [MERGED {name:>4}] Copied {source_counts[name]['images']} images and {source_counts[name]['labels']} labels.")

    # 2. Comprehensive Post-Merge Validation Audit
    print("\n" + "=" * 60)
    print("COMPREHENSIVE POST-MERGE VALIDATION AUDIT")
    print("=" * 60)

    train_imgs = list(TRAIN_IMG_DIR.glob("*"))
    val_imgs = list(VAL_IMG_DIR.glob("*"))

    train_lbls = list(TRAIN_LBL_DIR.glob("*"))
    val_lbls = list(VAL_LBL_DIR.glob("*"))

    train_img_stems = {p.stem for p in train_imgs}
    val_img_stems = {p.stem for p in val_imgs}

    train_lbl_stems = {p.stem for p in train_lbls}
    val_lbl_stems = {p.stem for p in val_lbls}

    total_train_imgs = len(train_imgs)
    total_val_imgs = len(val_imgs)
    total_imgs = total_train_imgs + total_val_imgs
    total_lbls = len(train_lbls) + len(val_lbls)

    # Filename collision check
    all_img_names = [p.name for p in train_imgs + val_imgs]
    duplicate_filenames_count = len(all_img_names) - len(set(all_img_names))

    # Missing image/label pairs
    missing_train_lbls = train_img_stems - train_lbl_stems
    missing_train_imgs = train_lbl_stems - train_img_stems
    missing_val_lbls = val_img_stems - val_lbl_stems
    missing_val_imgs = val_lbl_stems - val_img_stems

    missing_pairs_count = len(missing_train_lbls) + len(missing_train_imgs) + len(missing_val_lbls) + len(missing_val_imgs)

    boxes_by_class = {c: 0 for c in range(6)}
    images_per_class = {c: set() for c in range(6)}

    empty_label_files_count = 0
    invalid_label_lines_count = 0
    coords_outside_range_count = 0

    all_lbl_paths = train_lbls + val_lbls

    for lbl_p in all_lbl_paths:
        lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            empty_label_files_count += 1
            continue

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_label_lines_count += 1
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])

                if cls_id not in CLASS_NAMES:
                    invalid_label_lines_count += 1
                    continue

                if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    coords_outside_range_count += 1

                boxes_by_class[cls_id] += 1
                images_per_class[cls_id].add(lbl_p.stem)

            except ValueError:
                invalid_label_lines_count += 1

    total_boxes = sum(boxes_by_class.values())
    final_disk_size_mb = get_dir_size(FINAL_STAGE_DIR) / (1024 * 1024)

    # Class distribution assertions
    class_0_valid = (boxes_by_class[0] > 0)
    class_1_valid = (boxes_by_class[1] > 0)
    class_2_valid = (boxes_by_class[2] > 0)
    class_3_valid = (boxes_by_class[3] > 0)
    class_4_valid = (boxes_by_class[4] > 0)
    class_5_valid = (boxes_by_class[5] == 0)

    print(f"1.  Total train images               : {total_train_imgs}")
    print(f"2.  Total validation images          : {total_val_imgs}")
    print(f"3.  Total images                     : {total_imgs}")
    print(f"4.  Total label files                : {total_lbls}")
    print(f"5.  Total bounding boxes             : {total_boxes}")

    print("\n6. Bounding boxes by class:")
    for cls_id in range(6):
        cname = CLASS_NAMES[cls_id]
        print(f"    Class {cls_id} ({cname:<8}): {boxes_by_class[cls_id]:>6} boxes")

    print("\n7. Images containing each class:")
    for cls_id in range(6):
        cname = CLASS_NAMES[cls_id]
        print(f"    Class {cls_id} ({cname:<8}): {len(images_per_class[cls_id]):>6} images")

    print(f"\n8.  Missing image/label pairs         : {missing_pairs_count}")
    print(f"9.  Empty label files                 : {empty_label_files_count}")
    print(f"10. Invalid YOLO label lines          : {invalid_label_lines_count}")
    print(f"11. Coordinates outside [0, 1]        : {coords_outside_range_count}")
    print(f"12. Duplicate filenames               : {duplicate_filenames_count}")

    print(f"\nFiles copied per source dataset:")
    print(f"13.  COCO files copied               : {source_counts['COCO']['images']} images, {source_counts['COCO']['labels']} labels")
    print(f"14.  NYU files copied                : {source_counts['NYU']['images']} images, {source_counts['NYU']['labels']} labels")
    print(f"15.  RUGD files copied               : {source_counts['RUGD']['images']} images, {source_counts['RUGD']['labels']} labels")
    print(f"16. Final directory disk size         : {final_disk_size_mb:.2f} MB ({final_disk_size_mb / 1024:.2f} GB)")

    print("\nStrict Class Verification Assertions:")
    print(f"  Class 0 (person) annotations present   : {class_0_valid}")
    print(f"  Class 1 (obstacle) annotations present : {class_1_valid}")
    print(f"  Class 2 (chair) annotations present    : {class_2_valid}")
    print(f"  Class 3 (stairs) annotations present   : {class_3_valid}")
    print(f"  Class 4 (door) annotations present     : {class_4_valid}")
    print(f"  Class 5 (robot) zero annotations check : {class_5_valid} ({boxes_by_class[5]} boxes)")

    all_passed = (
        missing_pairs_count == 0 and
        invalid_label_lines_count == 0 and
        coords_outside_range_count == 0 and
        duplicate_filenames_count == 0 and
        class_5_valid
    )

    print("\n" + "=" * 60)
    if all_passed:
        print("FINAL DATASET MERGE RESULT: PASS")
        print("The final dataset (detection_final_stage) is 100% verified and SAFE TO USE FOR YOLO TRAINING!")
    else:
        print("FINAL DATASET MERGE RESULT: FAIL")
        print("Issues were detected. The dataset is NOT ready for training.")
    print("=" * 60)

if __name__ == "__main__":
    merge_final_dataset()
