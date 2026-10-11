"""
evaluate.py  –  Step 4: Evaluate the Trained Model
====================================================
What this does:
  1. Loads the best saved model from model_saved/best_model.keras
     (or from model_saved/saved_model if the .keras file doesn't exist).
  2. Runs predictions on the Kaggle test set (dataset_split/test/).
  3. If a real-world test set exists (dataset_split/real_world_test/),
     runs predictions on that too.
  4. Prints per-class accuracy as counts (e.g. 3/4 correct) and percentages.
  5. Saves a confusion matrix image to results/.

Why does the Kaggle test set have low numbers?
  Only ~3–4 images per class remain after deduplication and the 70/15/15 split.
  This means ONE wrong prediction = 25% drop in a class.  That's why we also
  report on the phone-photo real-world test set — it's the honest measure.

Run with:  python evaluate.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pathlib import Path
from collections import defaultdict
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# ─── Configuration ──────────────────────────────────────────────────────────
SPLIT_DIR  = Path("dataset_split")
SAVE_DIR   = Path("model_saved")
RESULTS    = Path("results")
IMG_SIZE   = (224, 224)
BATCH_SIZE = 8
THRESHOLD  = 0.85   # confidence threshold

RESULTS.mkdir(exist_ok=True)

# ─── Load class names ────────────────────────────────────────────────────────
with open("class_names.json") as f:
    CLASS_NAMES = json.load(f)
N_CLASSES = len(CLASS_NAMES)

print("=" * 60)
print("  STEP 4: Evaluation")
print("=" * 60)
print(f"Classes: {CLASS_NAMES}")

# ─── Load model ──────────────────────────────────────────────────────────────
keras_path = SAVE_DIR / "best_model.keras"
saved_path = SAVE_DIR / "saved_model"

if keras_path.exists():
    print(f"\nLoading model from {keras_path}")
    model = keras.models.load_model(str(keras_path))
elif saved_path.exists():
    print(f"\nLoading model from {saved_path}")
    model = keras.models.load_model(str(saved_path))
else:
    raise FileNotFoundError(
        "No trained model found. Run train.py first."
    )

# ─── Helper: preprocess a single image ───────────────────────────────────────

def preprocess_image(path: Path) -> np.ndarray:
    """Load one image, resize to 224×224, and scale to [-1, 1]."""
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 preprocessing: scale from [0,255] to [-1,1]
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return arr  # shape: (224, 224, 3)


def predict_folder(subset_name: str) -> dict:
    """
    Run predictions on all images in dataset_split/<subset_name>/.
    Returns a dict with overall accuracy and per-class counts.
    """
    subset_path = SPLIT_DIR / subset_name
    if not subset_path.exists():
        return None

    # We'll collect true and predicted labels
    y_true = []
    y_pred = []
    y_conf = []  # confidence scores

    img_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for cls_dir in sorted(subset_path.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls_name = cls_dir.name
        if cls_name not in CLASS_NAMES:
            print(f"  Warning: unknown class folder '{cls_name}' — skipping")
            continue
        true_idx = CLASS_NAMES.index(cls_name)

        for img_path in sorted(cls_dir.iterdir()):
            if img_path.suffix.lower() not in img_ext:
                continue
            arr = preprocess_image(img_path)
            arr = np.expand_dims(arr, axis=0)  # add batch dimension: (1, 224, 224, 3)
            probs = model.predict(arr, verbose=0)[0]  # probabilities for each class
            pred_idx = int(np.argmax(probs))
            conf = float(probs[pred_idx])

            y_true.append(true_idx)
            y_pred.append(pred_idx)
            y_conf.append(conf)

    if not y_true:
        return None

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_conf = np.array(y_conf)

    overall_acc = float(np.mean(y_true == y_pred))

    # Per-class accuracy as count strings like "3/4 correct (75.0%)"
    per_class = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        mask = y_true == i
        if mask.sum() == 0:
            per_class[cls_name] = "no images"
            continue
        correct = int((y_pred[mask] == i).sum())
        total   = int(mask.sum())
        pct     = 100.0 * correct / total
        per_class[cls_name] = f"{correct}/{total} correct ({pct:.1f}%)"

    # Confusion matrix (N_CLASSES × N_CLASSES grid)
    # cm[i][j] = number of images with true class i predicted as class j
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    return {
        "overall_accuracy": overall_acc,
        "per_class": per_class,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_conf": y_conf,
        "n": len(y_true),
    }


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: Path):
    """Draw a colour-coded confusion matrix and save it as a PNG."""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels([f"₹{c}" if c != "no_note" else "no_note" for c in CLASS_NAMES],
                       rotation=45, ha="right")
    ax.set_yticklabels([f"₹{c}" if c != "no_note" else "no_note" for c in CLASS_NAMES])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # Write numbers inside each cell
    thresh = cm.max() / 2.0
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Confusion matrix saved → {save_path}")


# ─── Evaluate Kaggle test set ─────────────────────────────────────────────────
print("\n── Kaggle test set (dataset_split/test/) ──")
kaggle_results = predict_folder("test")
if kaggle_results:
    acc = kaggle_results["overall_accuracy"]
    print(f"  Overall accuracy: {acc:.1%}  ({int(acc*kaggle_results['n'])}/{kaggle_results['n']})")
    print("\n  Per-class results:")
    print("  ⚠  Remember: only ~3–4 images per class — one wrong = big % drop")
    for cls, result in kaggle_results["per_class"].items():
        label = f"₹{cls}" if cls != "no_note" else "no_note"
        print(f"    {label:>10}: {result}")
    plot_confusion_matrix(
        kaggle_results["confusion_matrix"],
        f"Kaggle Test Set  (acc={acc:.1%})\n⚠ Only 3–4 images/class — results are noisy",
        RESULTS / "confusion_matrix_kaggle.png",
    )
else:
    print("  No test images found — skipping.")

# ─── Evaluate real-world phone-photo test set ─────────────────────────────────
print("\n── Real-world test set (dataset_split/real_world_test/) ──")
rw_results = predict_folder("real_world_test")
if rw_results:
    acc = rw_results["overall_accuracy"]
    print(f"  Overall accuracy: {acc:.1%}  ({int(acc*rw_results['n'])}/{rw_results['n']})")
    print("\n  Per-class results:")
    for cls, result in rw_results["per_class"].items():
        label = f"₹{cls}" if cls != "no_note" else "no_note"
        print(f"    {label:>10}: {result}")
    plot_confusion_matrix(
        rw_results["confusion_matrix"],
        f"Real-World Test Set  (acc={acc:.1%})",
        RESULTS / "confusion_matrix_realworld.png",
    )
else:
    print("  No real-world test images found.")
    print("  Add phone photos to my_photos/real_world_test/<class>/")
    print("  and re-run split_dataset.py + evaluate.py.")

# ─── Save a text summary ─────────────────────────────────────────────────────
with open(RESULTS / "evaluation_summary.txt", "w", encoding="utf-8") as f:
    f.write("Indian Currency Classifier — Evaluation Summary\n")
    f.write("=" * 50 + "\n\n")
    if kaggle_results:
        f.write(f"Kaggle test set accuracy: {kaggle_results['overall_accuracy']:.1%}\n")
        f.write("NOTE: Only ~3-4 images per class — results are noisy.\n")
        f.write("Per-class (Kaggle):\n")
        for cls, r in kaggle_results["per_class"].items():
            f.write(f"  {cls}: {r}\n")
    f.write("\n")
    if rw_results:
        f.write(f"Real-world test set accuracy: {rw_results['overall_accuracy']:.1%}\n")
        f.write("Per-class (Real-world):\n")
        for cls, r in rw_results["per_class"].items():
            f.write(f"  {cls}: {r}\n")
    else:
        f.write("Real-world test set: not available yet.\n")

print(f"\nEvaluation summary saved → {RESULTS}/evaluation_summary.txt")
print("=" * 60)
print("  Evaluation complete. Next: python export_tfjs.py")
print("=" * 60)
