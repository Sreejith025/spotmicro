import os
import sys
import time
import zipfile
import urllib.request
from pathlib import Path

DATASET_DIR = Path(__file__).parent.resolve()

def download_file_resumable(url, destination_path, expected_size=None, max_retries=5):
    """Download a file with HTTP Range support for resumption and progress updates."""
    dest = Path(destination_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(1, max_retries + 1):
        existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0
        req = urllib.request.Request(url)
        if existing_bytes > 0:
            req.add_header('Range', f'bytes={existing_bytes}-')
            print(f"[RESUME] Resuming download for {dest.name} from {existing_bytes / (1024*1024):.2f} MB (Attempt {attempt}/{max_retries})")
        else:
            print(f"[START] Starting download for {dest.name} (Attempt {attempt}/{max_retries})")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                status_code = response.status
                total_length = response.headers.get('Content-Length')
                if total_length:
                    total_bytes = int(total_length) + (existing_bytes if status_code == 206 else 0)
                else:
                    total_bytes = expected_size

                mode = 'ab' if status_code == 206 else 'wb'
                if mode == 'wb':
                    existing_bytes = 0

                downloaded = existing_bytes
                start_time = time.time()
                last_print = time.time()

                with open(temp_path, mode) as f:
                    block_size = 1024 * 1024  # 1 MB blocks
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)

                        now = time.time()
                        if now - last_print >= 3.0 or (total_bytes and downloaded == total_bytes):
                            elapsed = now - start_time
                            speed_mb = ((downloaded - existing_bytes) / (1024 * 1024)) / elapsed if elapsed > 0 else 0
                            if total_bytes:
                                percent = (downloaded / total_bytes) * 100
                                print(f"[{dest.name}] {downloaded / (1024*1024):.2f} MB / {total_bytes / (1024*1024):.2f} MB ({percent:.1f}%) - {speed_mb:.2f} MB/s", flush=True)
                            else:
                                print(f"[{dest.name}] {downloaded / (1024*1024):.2f} MB downloaded - {speed_mb:.2f} MB/s", flush=True)
                            last_print = now

            # Rename temp file to final destination once complete
            if temp_path.exists():
                if dest.exists():
                    dest.unlink()
                temp_path.rename(dest)
                print(f"[SUCCESS] Download completed: {dest.name} ({dest.stat().st_size / (1024*1024):.2f} MB)")
                return True

        except Exception as e:
            print(f"[WARN] Attempt {attempt}/{max_retries} failed for {dest.name}: {e}")
            if attempt < max_retries:
                time.sleep(3)
            else:
                print(f"[ERROR] All {max_retries} attempts failed for {url}")
                return False

def extract_zip(zip_path, extract_to):
    print(f"[EXTRACTING] Extracting {zip_path.name} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[SUCCESS] Extracted {zip_path.name}")

def download_rugd():
    print("\n" + "="*60)
    print("STAGE 1: DOWNLOADING RUGD (Robot Unstructured Ground Driving)")
    print("="*60)
    rugd_dir = DATASET_DIR / "rugd"
    zip_path = rugd_dir / "RUGD_frames-with-annotations.zip"
    url = "http://rugd.vision/data/RUGD_frames-with-annotations.zip"
    
    if zip_path.exists() and zip_path.stat().st_size >= 5600000000:
        print(f"[SKIP] RUGD archive already downloaded ({zip_path.stat().st_size / (1024*1024):.2f} MB)")
    else:
        success = download_file_resumable(url, zip_path, expected_size=5671298594)
        if not success:
            print("[FAIL] RUGD download did not complete successfully.")
            return False

    # Extract if not already extracted
    extracted_dir = rugd_dir / "RUGD_frames-with-annotations"
    if not extracted_dir.exists():
        extract_zip(zip_path, rugd_dir)
    else:
        print(f"[SKIP] RUGD already extracted at {extracted_dir}")
    return True

def download_nyu_depth_v2():
    print("\n" + "="*60)
    print("STAGE 1: DOWNLOADING NYU DEPTH V2 LABELED DATASET")
    print("="*60)
    nyu_dir = DATASET_DIR / "nyu_depth_v2"
    mat_path = nyu_dir / "nyu_depth_v2_labeled.mat"
    url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    
    if mat_path.exists() and mat_path.stat().st_size >= 2800000000:
        print(f"[SKIP] NYU Depth V2 already downloaded ({mat_path.stat().st_size / (1024*1024):.2f} MB)")
        return True
    
    return download_file_resumable(url, mat_path, expected_size=2810137344)

def download_coco_subset():
    print("\n" + "="*60)
    print("STAGE 1: DOWNLOADING COCO 2017 SUBSET (VAL + ANNOTATIONS)")
    print("="*60)
    coco_dir = DATASET_DIR / "coco_subset"
    val_zip = coco_dir / "val2017.zip"
    ann_zip = coco_dir / "annotations_trainval2017.zip"
    
    val_url = "http://images.cocodataset.org/zips/val2017.zip"
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    s1 = download_file_resumable(val_url, val_zip, expected_size=815581466)
    s2 = download_file_resumable(ann_url, ann_zip, expected_size=252907541)
    
    if s1 and not (coco_dir / "val2017").exists():
        extract_zip(val_zip, coco_dir)
    if s2 and not (coco_dir / "annotations").exists():
        extract_zip(ann_zip, coco_dir)
    return s1 and s2

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Stage 1 Dataset Downloader")
    parser.add_argument("--dataset", choices=["rugd", "nyu", "coco", "all"], default="all", help="Dataset to download")
    args = parser.parse_args()

    print(f"Starting Stage 1 Downloads in target directory: {DATASET_DIR}")
    
    if args.dataset in ["nyu", "all"]:
        download_nyu_depth_v2()
    
    if args.dataset in ["coco", "all"]:
        download_coco_subset()

    if args.dataset in ["rugd", "all"]:
        download_rugd()

    print("\n[DONE] Stage 1 download task finished!")
