"""
inspect_data.py  –  Step 1: Inspect the Dataset
================================================
This script does three things:
  1. Counts how many images exist in each class (Train + Test).
  2. Finds exact duplicate images using the MD5 hash algorithm
     (MD5 = a short "fingerprint" of a file's bytes; same fingerprint = same file).
  3. Tries to open every image with Pillow to catch corrupt / unreadable files.

Run it with:  python inspect_data.py
It does NOT modify any files inside the archive/ folder.
"""

import os
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from PIL import Image  # Pillow – for opening images

# ─── Configuration ──────────────────────────────────────────────────────────
ARCHIVE_DIR = Path("archive")   # the raw dataset folder (read-only)
SPLITS      = ["Train", "Test"] # the two sub-folders inside archive/

# Map the Kaggle folder names to the denomination numbers we actually want.
# 2Thousandnote is kept here just so we can report and skip it later.
CLASS_MAP = {
    "Tennote":       "10",
    "Twentynote":    "20",
    "Fiftynote":     "50",
    "1Hundrednote":  "100",
    "2Hundrednote":  "200",
    "5Hundrednote":  "500",
    "2Thousandnote": "2000",   # will be skipped during training
}

# Image file extensions we accept
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

# ─── Helpers ────────────────────────────────────────────────────────────────

def md5_of_file(filepath: Path) -> str:
    """Return the MD5 hash (hex string) of a file's raw bytes."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_image_ok(filepath: Path) -> bool:
    """Return True if Pillow can open the file without an error."""
    try:
        with Image.open(filepath) as img:
            img.verify()   # checks the file header without loading all pixels
        return True
    except Exception:
        return False


# ─── Main inspection ────────────────────────────────────────────────────────

def inspect():
    print("=" * 60)
    print("  STEP 1: Dataset Inspection")
    print("=" * 60)

    all_files: list[Path] = []       # every image file found
    counts = defaultdict(int)        # {class_name: count}
    corrupt: list[Path] = []         # files that could not be opened
    hash_to_files = defaultdict(list)  # {md5: [file, file, ...]}

    # Walk through Train/ and Test/ inside archive/
    for split in SPLITS:
        split_path = ARCHIVE_DIR / split
        if not split_path.exists():
            print(f"  WARNING: {split_path} does not exist – skipping.")
            continue

        for cls_dir in sorted(split_path.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name
            label    = CLASS_MAP.get(cls_name, cls_name)  # friendly denomination

            imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
            counts[label] += len(imgs)

            for img_path in imgs:
                all_files.append(img_path)
                h = md5_of_file(img_path)
                hash_to_files[h].append(img_path)

                if not is_image_ok(img_path):
                    corrupt.append(img_path)

    # ── Report: counts ───────────────────────────────────────────────────────
    print(f"\nTotal images found: {len(all_files)}")
    print("\nImages per class (both Train + Test combined):")
    print(f"  {'Class':>10}   {'Count':>5}")
    print(f"  {'-'*10}   {'-'*5}")
    for cls in sorted(counts.keys(), key=lambda x: int(x) if x.isdigit() else 9999):
        note = f"  {'-> SKIPPED (withdrawn)' if cls == '2000' else ''}"
        print(f"  {cls:>10}   {counts[cls]:>5}{note}")

    # ── Report: duplicates ───────────────────────────────────────────────────
    dupe_groups = {h: paths for h, paths in hash_to_files.items() if len(paths) > 1}
    total_dupes = sum(len(v) - 1 for v in dupe_groups.values())  # extra copies
    print(f"\nExact duplicates (MD5): {total_dupes} extra file(s) in {len(dupe_groups)} group(s)")
    for h, paths in dupe_groups.items():
        print(f"  MD5={h[:10]}… → {[str(p) for p in paths]}")

    # ── Report: corrupt ──────────────────────────────────────────────────────
    print(f"\nCorrupt / unreadable images: {len(corrupt)}")
    for p in corrupt:
        print(f"  {p}")

    # ── Save a JSON summary so other scripts can read it ────────────────────
    summary = {
        "total_images": len(all_files),
        "counts_by_class": dict(counts),
        "duplicate_groups": [[str(p) for p in v] for v in dupe_groups.values()],
        "corrupt_files": [str(p) for p in corrupt],
    }
    with open("inspect_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved inspect_report.json  (used by split_dataset.py)")
    print("=" * 60)


if __name__ == "__main__":
    inspect()
