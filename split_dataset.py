"""
split_dataset.py  –  Step 2: Deduplicate + Split the Dataset
=============================================================
What this script does:
  1. Reads all images from archive/ (Train + Test folders).
  2. Drops the 2000-rupee class (withdrawn from circulation).
  3. Removes exact duplicates (same MD5 hash) — keeps only one copy of each.
  4. Splits the remaining images into train / val / test  (70% / 15% / 15%)
     using a fixed random seed (42) so results are reproducible.
  5. Copies (never moves) the files into dataset_split/ which has sub-folders:
       dataset_split/train/<class>/
       dataset_split/val/<class>/
       dataset_split/test/<class>/
  6. If your phone photos exist in my_photos/, they are added:
       - my_photos/no_note/  →  split into train/val/test for the no_note class.
       - my_photos/<denom>/  →  added ONLY to train and val (never test).
       A separate real-world test set is kept in my_photos/real_world_test/.

Run with:  python split_dataset.py
Output:    dataset_split/   (safe to delete and re-run)
"""

import os
import shutil
import hashlib
import random
import json
from pathlib import Path
from collections import defaultdict

# ─── Configuration ──────────────────────────────────────────────────────────
ARCHIVE_DIR   = Path("archive")
MY_PHOTOS_DIR = Path("my_photos")         # your phone photos (optional)
OUT_DIR       = Path("dataset_split")     # output – created fresh each run
SPLITS        = ["Train", "Test"]
SKIP_CLASS    = "2Thousandnote"           # withdrawn; we don't train on ₹2000
RANDOM_SEED   = 42
TRAIN_FRAC    = 0.70
VAL_FRAC      = 0.15
# TEST_FRAC   = 0.15  (the remainder)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kaggle folder name → denomination string (used as our class folder name)
CLASS_MAP = {
    "Tennote":      "10",
    "Twentynote":   "20",
    "Fiftynote":    "50",
    "1Hundrednote": "100",
    "2Hundrednote": "200",
    "5Hundrednote": "500",
}
# Note classes we want (from Kaggle). no_note comes from phone photos only.
DENOM_CLASSES = list(CLASS_MAP.values())

# ─── Helpers ────────────────────────────────────────────────────────────────

def md5_of_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path):
    """Copy src → dst, creating parent directories as needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def split_list(items: list, train_f: float, val_f: float, seed: int):
    """
    Split a list into three parts: train / val / test.
    train_f and val_f are fractions (e.g. 0.70 and 0.15).
    The remainder goes to test.
    """
    rng = random.Random(seed)
    lst = items.copy()
    rng.shuffle(lst)
    n = len(lst)
    n_train = int(n * train_f)
    n_val   = int(n * val_f)
    # test gets the rest (ensures no image is lost due to rounding)
    return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  STEP 2: Deduplicate + Split Dataset")
    print("=" * 60)

    # ── 1. Collect all unique images from archive/, grouped by class ─────────
    # We ignore Kaggle's original Train/Test split and re-pool everything.
    class_images: dict[str, list[Path]] = defaultdict(list)
    seen_hashes: set[str] = set()   # to detect and skip duplicates
    dup_count = 0

    for split in SPLITS:
        split_path = ARCHIVE_DIR / split
        if not split_path.exists():
            continue
        for kaggle_cls_dir in sorted(split_path.iterdir()):
            if not kaggle_cls_dir.is_dir():
                continue
            if kaggle_cls_dir.name == SKIP_CLASS:
                # Skip the 2000-rupee class entirely
                print(f"  Skipping class: {kaggle_cls_dir.name}")
                continue
            label = CLASS_MAP.get(kaggle_cls_dir.name)
            if label is None:
                print(f"  Unknown class folder: {kaggle_cls_dir.name} — skipping")
                continue

            for img_path in sorted(kaggle_cls_dir.iterdir()):
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                h = md5_of_file(img_path)
                if h in seen_hashes:
                    dup_count += 1
                    continue   # skip duplicate
                seen_hashes.add(h)
                class_images[label].append(img_path)

    print(f"\nDuplicates removed: {dup_count}")
    print("\nUnique images per class (after deduplication):")
    for cls in sorted(DENOM_CLASSES, key=int):
        print(f"  ₹{cls}: {len(class_images[cls])} images")

    # ── 2. Split each note class 70/15/15 ────────────────────────────────────
    print(f"\nSplitting with seed={RANDOM_SEED}  (train 70% / val 15% / test 15%)…")

    # Clean output directory so we start fresh
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    split_report = {}
    for cls, imgs in class_images.items():
        train, val, test = split_list(imgs, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED)
        split_report[cls] = {"train": len(train), "val": len(val), "test": len(test)}
        for subset, files in [("train", train), ("val", val), ("test", test)]:
            for src in files:
                dst = OUT_DIR / subset / cls / src.name
                copy_file(src, dst)

    print("\nSplit results (Kaggle images only):")
    print(f"  {'Class':>6}  {'Train':>5}  {'Val':>5}  {'Test':>5}")
    for cls in sorted(DENOM_CLASSES, key=int):
        r = split_report.get(cls, {"train": 0, "val": 0, "test": 0})
        print(f"  {cls:>6}  {r['train']:>5}  {r['val']:>5}  {r['test']:>5}")

    # ── 3. Add phone photos (if the my_photos/ folder exists) ────────────────
    phone_classes_found = []
    phone_classes_missing = []

    if MY_PHOTOS_DIR.exists():
        print(f"\nFound {MY_PHOTOS_DIR}/ — adding phone photos…")

        for cls_dir in sorted(MY_PHOTOS_DIR.iterdir()):
            if not cls_dir.is_dir():
                continue
            cls_name = cls_dir.name  # e.g. "no_note", "10", "500" …

            imgs = [f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
            if len(imgs) == 0:
                continue
            phone_classes_found.append(cls_name)

            if cls_name == "no_note":
                # Split no_note photos into train/val/test just like Kaggle images
                train, val, test = split_list(imgs, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED)
                for subset, files in [("train", train), ("val", val), ("test", test)]:
                    for src in files:
                        dst = OUT_DIR / subset / "no_note" / src.name
                        copy_file(src, dst)
                print(f"    no_note: {len(train)} train / {len(val)} val / {len(test)} test")
            elif cls_name == "real_world_test":
                # Sub-folders inside real_world_test/ go to test/ directly
                for sub in cls_dir.iterdir():
                    if not sub.is_dir():
                        continue
                    for img_path in sub.iterdir():
                        if img_path.suffix.lower() in IMG_EXTS:
                            dst = OUT_DIR / "real_world_test" / sub.name / img_path.name
                            copy_file(img_path, dst)
                print(f"    real_world_test/: copied to dataset_split/real_world_test/")
            elif cls_name in DENOM_CLASSES:
                # Note photos from phone: add to train and val only (not test)
                train, val, _ = split_list(imgs, TRAIN_FRAC, VAL_FRAC, RANDOM_SEED)
                for subset, files in [("train", train), ("val", val)]:
                    for src in files:
                        dst = OUT_DIR / subset / cls_name / src.name
                        copy_file(src, dst)
                print(f"    {cls_name}: {len(train)} train / {len(val)} val (not added to test)")

        # Check for missing classes
        all_expected = DENOM_CLASSES + ["no_note"]
        for cls in all_expected:
            cls_dir = MY_PHOTOS_DIR / cls
            if not cls_dir.exists() or not any(True for _ in cls_dir.iterdir()):
                phone_classes_missing.append(cls)
    else:
        print(f"\n  my_photos/ folder NOT found.")
        phone_classes_missing = DENOM_CLASSES + ["no_note"]

    if phone_classes_missing:
        print(f"\n  ⚠  Phone photos MISSING for: {phone_classes_missing}")
        print("     The no_note class requires phone photos to work.")
        print("     Other missing classes mean less real-world training data.")
        print("     Add photos to my_photos/<class>/ and re-run this script.")

    # ── 4. Final count ───────────────────────────────────────────────────────
    print("\nFinal dataset_split/ image counts:")
    for subset in ["train", "val", "test", "real_world_test"]:
        subset_path = OUT_DIR / subset
        if not subset_path.exists():
            continue
        print(f"  {subset}/")
        for cls_dir in sorted(subset_path.iterdir()):
            if cls_dir.is_dir():
                n = len([f for f in cls_dir.iterdir() if f.is_file()])
                print(f"    {cls_dir.name}: {n}")

    # Save a split summary JSON for train.py to read
    summary = {
        "split_dir": str(OUT_DIR),
        "classes": DENOM_CLASSES,
        "has_no_note": "no_note" in phone_classes_found,
        "has_real_world_test": (OUT_DIR / "real_world_test").exists(),
    }
    with open("split_report.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSaved split_report.json")
    print("=" * 60)
    print("  Split complete. Next: python train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
