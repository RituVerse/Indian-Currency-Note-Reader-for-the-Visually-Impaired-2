"""
export_tfjs.py  –  Step 5: Export Model to TensorFlow.js
=========================================================
What this does:
  Converts the trained Keras model (SavedModel format) into TensorFlow.js
  format (a model.json file + binary weight shards).  This allows the model
  to run DIRECTLY IN THE BROWSER — no Python needed by the end user.

The output goes into the model/ folder.

Run with:  python export_tfjs.py

If this fails with a Keras 3 / TF version error, the script will automatically
try the tf_keras fallback path and print the exact error for you.
"""

import os
import sys
import json
from pathlib import Path

SAVE_DIR   = Path("model_saved")
TFJS_OUT   = Path("model")
IMG_SIZE   = (224, 224)

print("=" * 60)
print("  STEP 5: Export to TensorFlow.js")
print("=" * 60)

# ── Method 1: tensorflowjs_converter (command-line tool) ─────────────────────
# This is the standard way.  It reads a SavedModel and writes model.json + weights.

saved_model_path = SAVE_DIR / "saved_model"
keras_path       = SAVE_DIR / "best_model.keras"

if not saved_model_path.exists() and not keras_path.exists():
    print("ERROR: No model found in model_saved/. Run train.py first.")
    sys.exit(1)

TFJS_OUT.mkdir(exist_ok=True)

def try_converter():
    """Try the tensorflowjs_converter CLI tool."""
    import subprocess
    src = str(saved_model_path) if saved_model_path.exists() else str(keras_path)
    fmt = "tf_saved_model" if saved_model_path.exists() else "keras"

    cmd = [
        sys.executable, "-m", "tensorflowjs.converters.converter",
        "--input_format", fmt,
        "--output_format", "tfjs_graph_model",
        src,
        str(TFJS_OUT),
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ Conversion successful!")
        return True
    else:
        print("  ✗ Conversion failed. Error output:")
        print(result.stderr[-2000:])  # last 2000 chars to avoid flooding
        return False


def try_python_api():
    """Fallback: use the tensorflowjs Python API directly."""
    try:
        import tensorflowjs as tfjs
        from tensorflow import keras

        print("\nFallback: trying tensorflowjs Python API…")

        if keras_path.exists():
            model = keras.models.load_model(str(keras_path))
        else:
            model = keras.models.load_model(str(saved_model_path))

        tfjs.converters.save_keras_model(model, str(TFJS_OUT))
        print("  ✓ Conversion successful via Python API!")
        return True
    except Exception as e:
        print(f"  ✗ Python API also failed: {e}")
        return False


# Try the CLI converter first, then fall back to the Python API
success = try_converter()
if not success:
    success = try_python_api()

if not success:
    print("""
──────────────────────────────────────────────────────────
MANUAL FALLBACK INSTRUCTIONS
──────────────────────────────────────────────────────────
Both automatic methods failed. You have two options:

Option A: Use pip to install a matching version
  pip install tensorflowjs==4.22.0  (match your TF version)
  Then re-run: python export_tfjs.py

Option B: Use the command line directly
  tensorflowjs_converter \\
    --input_format=tf_saved_model \\
    model_saved/saved_model \\
    model/

Copy the error message above and paste it into the project
README under "Known Issues" so the next helper can see it.
──────────────────────────────────────────────────────────
""")
    sys.exit(1)

# ── Verify the output ─────────────────────────────────────────────────────────
model_json = TFJS_OUT / "model.json"
if model_json.exists():
    print(f"\nFiles in model/ folder:")
    for f in sorted(TFJS_OUT.iterdir()):
        size_kb = f.stat().st_size // 1024
        print(f"  {f.name}  ({size_kb} KB)")
else:
    print("\nWARNING: model.json not found in model/ — conversion may have failed silently.")

# ── Also save the class names inside model/ for the web app ──────────────────
with open("class_names.json") as f:
    cls = json.load(f)
with open(TFJS_OUT / "class_names.json", "w") as f:
    json.dump(cls, f, indent=2)
print(f"\nCopied class_names.json → model/class_names.json")

print("=" * 60)
print("  Export complete. Next: open index.html via: python serve.py")
print("=" * 60)
