"""
train.py  –  Step 3: Train the MobileNetV2 Model
=================================================
What this does:
  1. Loads images from dataset_split/ (created by split_dataset.py).
  2. Applies data augmentation (rotation, brightness, zoom — NO flips).
  3. Phase 1 – Frozen base: only our new top layers learn (fast).
  4. Phase 2 – Fine-tune: unfreeze the last 20 layers of MobileNetV2 (slower,
     but the model learns India-specific note features instead of just ImageNet).
  5. Saves the best model (by validation accuracy) as model_saved/.
  6. Saves training curves (accuracy & loss plots) in results/.

Why MobileNetV2?
  - Pre-trained on 1.4 million images (ImageNet) → already knows edges, shapes,
    textures.  We just teach it the difference between ₹10, ₹20, ... ₹500.
  - Very small and fast → runs in a browser after converting to TF.js.

Run with:  python train.py
Outputs:   model_saved/   results/training_curves.png
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")           # keep the output clean

# Tell TensorFlow to use only the CPUs (no GPU needed for this project)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # suppress TF startup noise

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (no window needed)
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────
SPLIT_DIR  = Path("dataset_split")   # from split_dataset.py
SAVE_DIR   = Path("model_saved")     # where the trained model is stored
RESULTS    = Path("results")         # where plots are saved

IMG_SIZE   = (224, 224)   # MobileNetV2 input size
BATCH_SIZE = 8            # small batch = safer on low-RAM machines
EPOCHS_FROZEN  = 15       # Phase 1: train only the top layers
EPOCHS_FINETUNE= 15       # Phase 2: fine-tune last 20 layers
FINE_TUNE_AT   = -20      # unfreeze from this layer onwards (last 20 layers)
LR_FROZEN      = 1e-3     # learning rate for Phase 1
LR_FINETUNE    = 1e-5     # much lower LR for Phase 2 (avoid destroying pretrained weights)
SEED           = 42

# ─── Discover classes ────────────────────────────────────────────────────────
# Classes = sub-folder names inside dataset_split/train/
train_dir = SPLIT_DIR / "train"
CLASS_NAMES = sorted([d.name for d in train_dir.iterdir() if d.is_dir()],
                     key=lambda x: (int(x) if x.isdigit() else 9999, x))
N_CLASSES = len(CLASS_NAMES)

print("=" * 60)
print("  STEP 3: Training")
print("=" * 60)
print(f"Classes ({N_CLASSES}): {CLASS_NAMES}")
print(f"Image size: {IMG_SIZE}, Batch size: {BATCH_SIZE}")

RESULTS.mkdir(exist_ok=True)
SAVE_DIR.mkdir(exist_ok=True)

# Save class order to a JSON file — the web app needs this to map index → denomination
with open("class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f, indent=2)
print(f"Class names saved → class_names.json")

# ─── Data loading ────────────────────────────────────────────────────────────
# `image_dataset_from_directory` reads images, resizes them to 224×224,
# and returns batches of (image, label) pairs automatically.

def make_dataset(subset_name: str, augment: bool = False):
    """Load a dataset split.  subset_name = 'train', 'val', or 'test'."""
    ds = keras.utils.image_dataset_from_directory(
        SPLIT_DIR / subset_name,
        labels       = "inferred",        # labels come from sub-folder names
        label_mode   = "categorical",     # one-hot encoded (e.g. [0,0,1,0,0,0,0])
        class_names  = CLASS_NAMES,       # fixes the order
        image_size   = IMG_SIZE,
        batch_size   = BATCH_SIZE,
        shuffle      = (subset_name == "train"),
        seed         = SEED,
    )
    return ds

train_ds_raw = make_dataset("train", augment=True)
val_ds_raw   = make_dataset("val",   augment=False)
test_ds_raw  = make_dataset("test",  augment=False)

# ─── Augmentation pipeline ───────────────────────────────────────────────────
# We apply random augmentation ONLY to the training set to make the model
# more robust to real-world conditions (different angles, lighting).
# No flips — they would mirror the note text/design.

augmentation = keras.Sequential([
    layers.RandomRotation(factor=0.04),          # ±15° (0.04 × 360° ≈ 15°)
    layers.RandomZoom(height_factor=0.10),        # ±10% zoom
    layers.RandomBrightness(factor=0.20),         # ±20% brightness
], name="augmentation")

# MobileNetV2 expects pixel values in [-1, 1]. This layer does that rescaling.
preprocess = layers.Lambda(
    lambda x: tf.keras.applications.mobilenet_v2.preprocess_input(x),
    name="preprocess"
)

def prepare(ds, augment=False):
    """
    Apply preprocessing (and optionally augmentation) then cache and prefetch.
    .cache()    = keeps the dataset in RAM after the first epoch → faster.
    .prefetch() = loads the next batch while the GPU/CPU trains on this one.
    """
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (preprocess(x), y),
                num_parallel_calls=tf.data.AUTOTUNE)
    return ds.cache().prefetch(tf.data.AUTOTUNE)

train_ds = prepare(train_ds_raw, augment=True)
val_ds   = prepare(val_ds_raw,   augment=False)
test_ds  = prepare(test_ds_raw,  augment=False)

# ─── Build the model ─────────────────────────────────────────────────────────
print("\nBuilding MobileNetV2 model…")

# Load MobileNetV2 pre-trained on ImageNet, WITHOUT its original top classifier.
# include_top=False means we get just the feature extractor part.
base_model = MobileNetV2(
    input_shape = IMG_SIZE + (3,),   # 224×224×3 (RGB)
    include_top = False,             # remove ImageNet's 1000-class head
    weights     = "imagenet",        # use pre-trained weights
)
base_model.trainable = False  # Phase 1: freeze all base layers

# Build our custom classification head on top
inputs = keras.Input(shape=IMG_SIZE + (3,))
x = base_model(inputs, training=False)              # run through frozen base
x = layers.GlobalAveragePooling2D()(x)             # flatten spatial dimensions
x = layers.Dense(128, activation="relu")(x)        # a small fully-connected layer
x = layers.Dropout(0.3)(x)                         # randomly zero 30% neurons = less overfitting
outputs = layers.Dense(N_CLASSES, activation="softmax")(x)  # final prediction

model = keras.Model(inputs, outputs, name="indian_currency_classifier")
model.summary()

# ─── Callbacks ───────────────────────────────────────────────────────────────
# These run automatically after each epoch:

# Save the weights whenever val_accuracy improves
checkpoint = ModelCheckpoint(
    filepath  = str(SAVE_DIR / "best_model.keras"),
    monitor   = "val_accuracy",
    save_best_only= True,
    verbose   = 1,
)

# Stop early if val_accuracy stops improving for 8 epochs in a row
early_stop = EarlyStopping(
    monitor  = "val_accuracy",
    patience = 8,
    restore_best_weights = True,
    verbose  = 1,
)

# Reduce learning rate if val_loss is stuck
reduce_lr = ReduceLROnPlateau(
    monitor  = "val_loss",
    factor   = 0.5,         # halve the LR
    patience = 4,
    min_lr   = 1e-7,
    verbose  = 1,
)

# ─── Phase 1: Train only the new top layers ───────────────────────────────────
print("\n── Phase 1: Training top layers (base frozen) ──")
model.compile(
    optimizer = keras.optimizers.Adam(LR_FROZEN),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

history1 = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs    = EPOCHS_FROZEN,
    callbacks = [checkpoint, early_stop, reduce_lr],
)

# ─── Phase 2: Fine-tune the last 20 layers of MobileNetV2 ────────────────────
print("\n── Phase 2: Fine-tuning last 20 base layers ──")

# Make all layers trainable first, then freeze everything EXCEPT the last 20
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:   # freeze all except last 20
    layer.trainable = False

# Count trainable layers for transparency
n_trainable = sum(1 for l in base_model.layers if l.trainable)
print(f"  Trainable base layers: {n_trainable} / {len(base_model.layers)}")

# Recompile with a much lower learning rate to avoid destroying the pretrained weights
model.compile(
    optimizer = keras.optimizers.Adam(LR_FINETUNE),
    loss      = "categorical_crossentropy",
    metrics   = ["accuracy"],
)

# Reset early-stop patience counter
early_stop2 = EarlyStopping(
    monitor  = "val_accuracy",
    patience = 10,
    restore_best_weights = True,
    verbose  = 1,
)

history2 = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs    = EPOCHS_FROZEN + EPOCHS_FINETUNE,
    initial_epoch = len(history1.history["accuracy"]),   # continue from where we stopped
    callbacks = [checkpoint, early_stop2, reduce_lr],
)

# ─── Save final model as TF SavedModel format (needed for TF.js export) ──────
model.save(str(SAVE_DIR / "saved_model"))
print(f"\nModel saved to {SAVE_DIR}/saved_model")

# ─── Plot training curves ─────────────────────────────────────────────────────
# Combine Phase 1 + Phase 2 histories into one long history
def combine(h1, h2, key):
    return h1.history[key] + h2.history[key]

epochs_total = list(range(1, len(combine(history1, history2, "accuracy")) + 1))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Accuracy plot
ax1.plot(epochs_total, combine(history1, history2, "accuracy"),     label="Train Acc")
ax1.plot(epochs_total, combine(history1, history2, "val_accuracy"), label="Val Acc")
ax1.axvline(len(history1.history["accuracy"]), color="gray", linestyle="--", label="Fine-tune start")
ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)

# Loss plot
ax2.plot(epochs_total, combine(history1, history2, "loss"),     label="Train Loss")
ax2.plot(epochs_total, combine(history1, history2, "val_loss"), label="Val Loss")
ax2.axvline(len(history1.history["loss"]), color="gray", linestyle="--", label="Fine-tune start")
ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend(); ax2.grid(True)

plt.suptitle("Indian Currency Classifier – Training Curves", fontsize=14)
plt.tight_layout()
curve_path = RESULTS / "training_curves.png"
plt.savefig(curve_path, dpi=150)
print(f"Training curves saved → {curve_path}")

print("=" * 60)
print("  Training complete. Next: python evaluate.py")
print("=" * 60)
