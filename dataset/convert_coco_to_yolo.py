import json
import os
import random
import shutil
from pathlib import Path

# Base Paths
BASE_DIR = Path(r"D:\spotmicro")
COCO_IMG_DIR = BASE_DIR / "dataset" / "public" / "coco" / "val2017"
COCO_ANN_FILE = BASE_DIR / "dataset" / "coco_subset" / "annotations" / "instances_val2017.json"
OUTPUT_DIR = BASE_DIR / "dataset" / "detection_coco_stage"

# Destination directories
TRAIN_IMG_DIR = OUTPUT_DIR / "images" / "train"
VAL_IMG_DIR = OUTPUT_DIR / "images" / "val"
TRAIN_LBL_DIR = OUTPUT_DIR / "labels" / "train"
VAL_LBL_DIR = OUTPUT_DIR / "labels" / "val"

# Category Mapping: COCO ID -> YOLO Class ID
# person (COCO ID 1) -> YOLO class 0
# chair (COCO ID 62) -> YOLO class 2
CATEGORY_MAPPING = {
    1: 0,   # person -> 0
    62: 2,  # chair  -> 2
}

SPLIT_RATIO = 0.8  # 80% train, 20% validation
RANDOM_SEED = 42

def convert_coco_to_yolo():
    print("=" * 60)
    print("COCO to YOLO Bounding Box Converter for SpotMicro")
    print("=" * 60)

    # Ensure output directories exist
    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO annotations from: {COCO_ANN_FILE}")
    with open(COCO_ANN_FILE, "r", encoding="utf-8") as f:
        coco_data = json.load(f)

    # Map image ID -> metadata
    images_by_id = {img["id"]: img for img in coco_data["images"]}

    # Collect target bounding boxes for each image
    image_to_boxes = {}

    for ann in coco_data["annotations"]:
        cat_id = ann.get("category_id")
        if cat_id not in CATEGORY_MAPPING:
            continue

        yolo_class_id = CATEGORY_MAPPING[cat_id]
        img_id = ann["image_id"]
        bbox = ann["bbox"]  # COCO format: [x_min, y_min, box_w, box_h]

        if img_id not in images_by_id:
            continue

        img_info = images_by_id[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]

        x_min, y_min, box_w, box_h = bbox

        if box_w <= 0 or box_h <= 0 or img_w <= 0 or img_h <= 0:
            continue

        # Convert to normalized YOLO format: x_center, y_center, width, height
        x_center = (x_min + box_w / 2.0) / img_w
        y_center = (y_min + box_h / 2.0) / img_h
        norm_w = box_w / img_w
        norm_h = box_h / img_h

        # Clamp values within bounds [0.0, 1.0]
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        norm_w = max(0.0, min(1.0, norm_w))
        norm_h = max(0.0, min(1.0, norm_h))

        yolo_line = f"{yolo_class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}"

        if img_id not in image_to_boxes:
            image_to_boxes[img_id] = []
        image_to_boxes[img_id].append((yolo_class_id, yolo_line))

    # Filter image IDs that exist on disk and contain at least one target box
    valid_image_ids = [
        img_id for img_id in image_to_boxes
        if (COCO_IMG_DIR / images_by_id[img_id]["file_name"]).exists()
    ]

    # Deterministic split: 80% train, 20% validation
    valid_image_ids.sort()
    random.seed(RANDOM_SEED)
    random.shuffle(valid_image_ids)

    num_total = len(valid_image_ids)
    num_train = int(num_total * SPLIT_RATIO)
    
    train_ids = set(valid_image_ids[:num_train])
    val_ids = set(valid_image_ids[num_train:])

    copied_images_count = 0
    person_boxes_count = 0
    chair_boxes_count = 0
    empty_images_count = 0

    print(f"Processing and copying {num_total} images containing target classes (person/chair)...")

    for img_id in valid_image_ids:
        img_info = images_by_id[img_id]
        file_name = img_info["file_name"]
        src_img_path = COCO_IMG_DIR / file_name

        boxes = image_to_boxes.get(img_id, [])

        if not boxes:
            empty_images_count += 1
            continue

        if img_id in train_ids:
            dst_img_dir = TRAIN_IMG_DIR
            dst_lbl_dir = TRAIN_LBL_DIR
        else:
            dst_img_dir = VAL_IMG_DIR
            dst_lbl_dir = VAL_LBL_DIR

        # Copy image file (preserving original files)
        dst_img_path = dst_img_dir / file_name
        shutil.copy2(src_img_path, dst_img_path)
        copied_images_count += 1

        # Write corresponding YOLO label file
        txt_filename = Path(file_name).stem + ".txt"
        dst_lbl_path = dst_lbl_dir / txt_filename

        with open(dst_lbl_path, "w", encoding="utf-8") as f_lbl:
            for cls_id, line in boxes:
                f_lbl.write(line + "\n")
                if cls_id == 0:
                    person_boxes_count += 1
                elif cls_id == 2:
                    chair_boxes_count += 1

    # Print final summary
    print("\n" + "=" * 60)
    print("CONVERSION SUMMARY")
    print("=" * 60)
    print(f"number of images copied        : {copied_images_count}")
    print(f"number of person boxes         : {person_boxes_count}")
    print(f"number of chair boxes          : {chair_boxes_count}")
    print(f"number of empty/no-label images: {empty_images_count}")
    print(f"number of train images         : {len(train_ids)}")
    print(f"number of validation images    : {len(val_ids)}")
    print("=" * 60)

if __name__ == "__main__":
    convert_coco_to_yolo()
