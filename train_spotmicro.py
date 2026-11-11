"""
SpotMicro Autonomous Navigation - YOLO Object Detection Training Script
------------------------------------------------------------------------
This script configures and executes the training pipeline for the SpotMicro
6-class object detection model (person, obstacle, chair, stairs, door, robot).

Hardware Configuration:
- GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- VRAM: 4 GB (Optimized with batch=4, workers=2, cache=False)
- OS: Windows 11
- Root: D:\spotmicro
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO

# Project Base Directory
BASE_DIR = Path(r"D:\spotmicro")
DATASET_DIR = BASE_DIR / "dataset" / "detection_final_stage"
YAML_PATH = DATASET_DIR / "data.yaml"
PROJECT_RUNS_DIR = BASE_DIR / "runs" / "detect"
EXPERIMENT_NAME = "spotmicro_yolo11n_v1"

def verify_environment_and_dataset():
    """Verify PyTorch CUDA environment and dataset paths before training."""
    print("=" * 65)
    print("SPOTMICRO YOLO TRAINING PIPELINE - PRE-FLIGHT ENVIRONMENT CHECK")
    print("=" * 65)

    # 1. PyTorch CUDA & GPU Checks
    cuda_available = torch.cuda.is_available()
    print(f"1. PyTorch CUDA Available : {cuda_available}")
    print(f"   PyTorch Version        : {torch.__version__}")
    print(f"   CUDA Toolkit Version   : {torch.version.cuda}")

    if not cuda_available:
        print("[WARNING] CUDA is NOT available! Training will run slowly on CPU.")
        gpu_name = "CPU Only"
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"2. GPU Device Name        : {gpu_name}")
        print(f"   Dedicated VRAM         : {vram_gb:.2f} GB")

    # 2. Verify Dataset YAML Existence
    print(f"\n3. Dataset YAML Config Path: {YAML_PATH}")
    if not YAML_PATH.exists():
        raise FileNotFoundError(f"Error: Training YAML file not found at: {YAML_PATH}")
    print("   [OK] YAML config file verified!")

    # 3. Verify Dataset Directory Structure
    train_img_dir = DATASET_DIR / "images" / "train"
    val_img_dir = DATASET_DIR / "images" / "val"
    train_lbl_dir = DATASET_DIR / "labels" / "train"
    val_lbl_dir = DATASET_DIR / "labels" / "val"

    required_dirs = [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]
    for d in required_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Error: Required dataset directory missing: {d}")
        file_count = len(list(d.glob("*")))
        print(f"   [OK] Verified directory: {d.relative_to(BASE_DIR)} ({file_count:,} files)")

    print("\n[SUCCESS] Environment and dataset pre-flight verification PASSED!")
    print("=" * 65)
    return gpu_name

def train():
    """Execute Ultralytics YOLO training with 4GB VRAM hardware optimizations."""
    gpu_name = verify_environment_and_dataset()

    print("\n" + "=" * 65)
    print("STARTING ULTRALYTICS YOLO TRAINING RUN")
    print("=" * 65)
    print(f"Model Architecture  : yolo11n.pt (Pretrained COCO Weights)")
    print(f"Dataset Config      : {YAML_PATH}")
    print(f"Image Resolution    : 640 x 640")
    print(f"Epochs              : 100")
    print(f"Batch Size          : 4 (VRAM-Optimized for 4GB RTX 3050)")
    print(f"Compute Device      : GPU 0 ({gpu_name})")
    print(f"Output Directory    : {PROJECT_RUNS_DIR / EXPERIMENT_NAME}")
    print("=" * 65 + "\n")

    # Load pretrained YOLO11 Nano model weights
    model = YOLO("yolo11n.pt")

    # Execute training with hardware-aware parameters
    results = model.train(
        data=str(YAML_PATH),          # Path to dataset YAML configuration
        imgsz=640,                    # Standard 640x640 resolution input size
        epochs=100,                   # Total training epochs
        batch=4,                      # Batch size 4 to prevent Out-Of-Memory (OOM) on 4GB VRAM
        device=0,                     # Use CUDA GPU device 0
        workers=2,                    # DataLoader worker threads (optimized for Windows)
        cache=False,                  # Disable RAM/VRAM image caching to save memory
        project=str(PROJECT_RUNS_DIR),# Root directory for runs: D:\spotmicro\runs\detect
        name=EXPERIMENT_NAME,         # Run experiment folder name
        exist_ok=True,                # Overwrite/continue existing experiment folder if rerun
        pretrained=True,              # Initialize with pretrained weights
        verbose=True,                 # Detailed training logs
    )

    print("\n" + "=" * 65)
    print("TRAINING RUN COMPLETE")
    print("=" * 65)
    print(f"Best Model Saved To: {PROJECT_RUNS_DIR / EXPERIMENT_NAME / 'weights' / 'best.pt'}")
    print(f"Last Model Saved To: {PROJECT_RUNS_DIR / EXPERIMENT_NAME / 'weights' / 'last.pt'}")
    print("=" * 65)

if __name__ == "__main__":
    train()
