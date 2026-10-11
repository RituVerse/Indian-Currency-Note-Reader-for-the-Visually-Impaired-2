"""
generate_no_note.py  –  Create synthetic "no_note" images
==========================================================
Since we have no phone photos of empty frames / hands / tables,
we generate synthetic images that look like:
  • Solid colour blocks (white, grey, black, beige, wood-tones)
  • Gaussian noise (random TV-static)
  • Smooth gradients (two-colour fades)
  • Blurred random blobs (soft shapes)
  • Random texture patches (fine noise convolved with a blur)

These images contain NO banknote features, so training on them
teaches the model to output "no_note" when it sees a blank scene.

IMPORTANT LIMITATION (added to README):
  The no_note class is entirely synthetic.  In real use the model
  may still misclassify a hand, wallet, or textured table as a note.
  This is much better than having NO no_note class (which would force
  the model to always name a denomination), but it is not the same as
  training on real "empty frame" photos.

Output:
  dataset_split/train/no_note/   (~24 images)
  dataset_split/val/no_note/     (~5 images)
  dataset_split/test/no_note/    (~6 images)
  Total: ~35 images

Run with:  python generate_no_note.py
"""

import os
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw

# ── Config ───────────────────────────────────────────────────────────────────
SPLIT_DIR  = Path("dataset_split")
SIZE       = (224, 224)          # must match MobileNetV2 input
TOTAL      = 35                  # total synthetic images to make
TRAIN_N    = 24
VAL_N      = 5
TEST_N     = 6                   # TRAIN_N + VAL_N + TEST_N = 35
SEED       = 42

rng = random.Random(SEED)
np_rng = np.random.default_rng(SEED)

# ── Helpers ──────────────────────────────────────────────────────────────────

def save(arr: np.ndarray, path: Path):
    """Save a uint8 numpy array as a JPEG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(str(path), "JPEG", quality=85)


def solid_colour(c):
    """224×224 image filled with a single RGB colour."""
    arr = np.full((224, 224, 3), c, dtype=np.uint8)
    return arr


def gradient(c1, c2):
    """Smooth left-to-right gradient between two colours."""
    arr = np.zeros((224, 224, 3), dtype=np.float32)
    for col in range(224):
        t = col / 223.0
        arr[:, col, :] = [c1[i] * (1 - t) + c2[i] * t for i in range(3)]
    return arr.astype(np.uint8)


def gaussian_noise():
    """Uniform random noise (looks like TV static)."""
    arr = np_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    return arr


def soft_noise():
    """Gaussian noise then heavily blurred → soft texture."""
    noise = np_rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(radius=12))
    return np.array(img)


def blobs():
    """
    Random pastel blobs on a plain background.
    Simulates an out-of-focus scene with coloured regions.
    """
    bg = rng.randint(180, 240)
    arr = np.full((224, 224, 3), bg, dtype=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for _ in range(rng.randint(3, 8)):
        x0 = rng.randint(0, 180)
        y0 = rng.randint(0, 180)
        x1 = x0 + rng.randint(30, 100)
        y1 = y0 + rng.randint(30, 100)
        col = (rng.randint(100, 240), rng.randint(100, 240), rng.randint(100, 240))
        draw.ellipse([x0, y0, x1, y1], fill=col)
    img = img.filter(ImageFilter.GaussianBlur(radius=15))
    return np.array(img)


def striped():
    """Horizontal coloured stripes (mimics a patterned surface)."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    stripe_h = rng.randint(8, 30)
    for row in range(0, 224, stripe_h):
        col = (rng.randint(80, 240), rng.randint(80, 240), rng.randint(80, 240))
        arr[row:row + stripe_h, :, :] = col
    return arr


def fine_texture():
    """Fine noise * blur = fabric / table texture look."""
    noise = np_rng.integers(120, 200, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(noise).filter(ImageFilter.GaussianBlur(radius=3))
    return np.array(img)


# ── Generate all images ───────────────────────────────────────────────────────

generators = []

# Solid colours — plain backgrounds
COLOURS = [
    (255, 255, 255),   # white (paper / wall)
    (220, 220, 220),   # light grey
    (180, 180, 180),   # grey
    (100, 100, 100),   # dark grey
    ( 30,  30,  30),   # near-black (dark room)
    (210, 180, 140),   # tan / beige (table)
    (139,  90,  43),   # brown (wood)
    (  0, 128,   0),   # green (cloth)
    (  0,   0, 128),   # dark blue (jeans / fabric)
    (255, 235, 200),   # skin tone approx
]
for c in COLOURS:
    generators.append(("solid", c))

# Gradients
GRAD_PAIRS = [
    ((255, 255, 255), (200, 200, 200)),
    ((180, 140, 100), (100, 70,  40)),
    ((200, 200, 255), (255, 200, 200)),
    ((50,  50,  50),  (150, 150, 150)),
    ((255, 220, 180), (200, 160, 120)),
]
for g in GRAD_PAIRS:
    generators.append(("gradient", g))

# Noise / texture
for _ in range(5):
    generators.append(("gaussian_noise", None))
for _ in range(5):
    generators.append(("soft_noise", None))
for _ in range(5):
    generators.append(("blobs", None))
for _ in range(3):
    generators.append(("striped", None))
for _ in range(3):
    generators.append(("fine_texture", None))

# Shuffle deterministically and take TOTAL
rng.shuffle(generators)
generators = generators[:TOTAL]

# ── Build images ─────────────────────────────────────────────────────────────
images = []
for i, (kind, param) in enumerate(generators):
    if kind == "solid":
        arr = solid_colour(param)
    elif kind == "gradient":
        arr = gradient(param[0], param[1])
    elif kind == "gaussian_noise":
        arr = gaussian_noise()
    elif kind == "soft_noise":
        arr = soft_noise()
    elif kind == "blobs":
        arr = blobs()
    elif kind == "striped":
        arr = striped()
    elif kind == "fine_texture":
        arr = fine_texture()
    else:
        arr = solid_colour((128, 128, 128))

    images.append((f"nonote_{i+1:03d}.jpg", arr))

# ── Split and write ───────────────────────────────────────────────────────────
assert len(images) == TOTAL

train_imgs = images[:TRAIN_N]
val_imgs   = images[TRAIN_N:TRAIN_N + VAL_N]
test_imgs  = images[TRAIN_N + VAL_N:]

counts = {"train": 0, "val": 0, "test": 0}
for subset, batch in [("train", train_imgs), ("val", val_imgs), ("test", test_imgs)]:
    out_dir = SPLIT_DIR / subset / "no_note"
    for fname, arr in batch:
        save(arr, out_dir / fname)
        counts[subset] += 1

print("Synthetic no_note images created:")
for k, v in counts.items():
    print(f"  dataset_split/{k}/no_note/  {v} images")
print(f"\nTotal: {sum(counts.values())} images")
print("\nIMPORTANT: These are synthetic (colour blocks, noise, gradients).")
print("The model may still misclassify a hand or textured surface as a note.")
print("This is a known limitation documented in the README.")
