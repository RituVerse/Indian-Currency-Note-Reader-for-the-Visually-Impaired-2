"""
export_tfjs_direct.py  –  Direct TF.js export bypassing broken tensorflowjs CLI
================================================================================
tensorflowjs 3.18.0 has two issues on Python 3.14 + NumPy 2.x:
  1. np.object / np.bool removed (patched in read_weights.py + write_weights.py)
  2. tensorflow_hub imports distutils which was removed in Python 3.12+

This script uses TF's own concrete function tracing to build a graph model
and writes model.json + weight shards manually — no tensorflowjs CLI needed.

We use the tensorflowjs.converters.convert_tf_saved_model path but mock out
the broken tensorflow_hub import so it never runs.
"""

import sys
import types
import os
import json
import struct
import numpy as np
from pathlib import Path

# ── Step 1: Mock out distutils and tensorflow_hub before any import ──────────
# This prevents the "No module named distutils" crash inside tensorflow_hub.
# We only need tensorflowjs for weight serialization, not hub features.

# Create a fake distutils.version module
fake_distutils = types.ModuleType("distutils")
fake_version   = types.ModuleType("distutils.version")
class LooseVersion:
    def __init__(self, v): self.v = v
    def __le__(self, other): return True
    def __lt__(self, other): return True
    def __ge__(self, other): return True
    def __gt__(self, other): return True
fake_version.LooseVersion = LooseVersion
fake_distutils.version = fake_version
sys.modules["distutils"]         = fake_distutils
sys.modules["distutils.version"] = fake_version

# Create a minimal fake tensorflow_hub module
fake_hub = types.ModuleType("tensorflow_hub")
fake_hub.load = lambda *a, **kw: None
fake_hub.KerasLayer = object
sys.modules["tensorflow_hub"] = fake_hub

# ── Step 2: Now import tensorflowjs safely ────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

SAVE_DIR = Path("model_saved")
TFJS_OUT = Path("model")
TFJS_OUT.mkdir(exist_ok=True)

print("=" * 60)
print("  STEP 5: Export to TensorFlow.js (direct method)")
print("=" * 60)

# Load the best trained model
keras_path = SAVE_DIR / "best_model.keras"
print(f"\nLoading model from {keras_path} …")
model = keras.models.load_model(str(keras_path))
print("Model loaded.")

# ── Step 3: Try tensorflowjs Python API with the mocked imports ───────────────
try:
    import tensorflowjs as tfjs
    print("\nAttempting tensorflowjs.converters.save_keras_model …")
    tfjs.converters.save_keras_model(model, str(TFJS_OUT))
    print("  Success via tensorflowjs Python API!")
    success = True
except Exception as e:
    print(f"  Failed: {e}")
    success = False

# ── Step 4: Fallback — manual weight export ────────────────────────────────────
if not success:
    print("\nFalling back to manual weight serialization …")

    # Get all trainable weights as numpy arrays
    weights_manifest = []
    all_weight_bytes = bytearray()
    weight_entries   = []

    for var in model.weights:
        arr   = var.numpy()
        name  = var.name
        shape = list(arr.shape)
        dtype_map = {
            "float32": "float32", "float16": "float16",
            "int32": "int32",     "int64": "float32",  # cast int64 → float32
        }
        dtype_str = dtype_map.get(str(arr.dtype), "float32")
        arr_cast  = arr.astype(np.float32) if dtype_str == "float32" else arr

        raw = arr_cast.tobytes()
        weight_entries.append({
            "name": name,
            "shape": shape,
            "dtype": dtype_str,
            "byteOffset": len(all_weight_bytes),
            "byteLength": len(raw),
        })
        all_weight_bytes.extend(raw)

    # Write the single weight shard file
    shard_name = "group1-shard1of1.bin"
    with open(TFJS_OUT / shard_name, "wb") as f:
        f.write(all_weight_bytes)

    # Build weights manifest
    weights_manifest = [{
        "paths": [shard_name],
        "weights": [
            {"name": e["name"], "shape": e["shape"], "dtype": e["dtype"]}
            for e in weight_entries
        ]
    }]

    # Build model topology JSON from Keras config
    model_config = model.to_json()
    model_topology = json.loads(model_config)

    model_json = {
        "format": "layers-model",
        "generatedBy": "manual-export",
        "convertedBy": "custom-exporter",
        "modelTopology": model_topology,
        "weightsManifest": weights_manifest,
    }

    with open(TFJS_OUT / "model.json", "w") as f:
        json.dump(model_json, f)

    print(f"  Manual export written to {TFJS_OUT}/")
    success = True

# ── Step 5: Copy class names ──────────────────────────────────────────────────
with open("class_names.json") as f:
    cls = json.load(f)
with open(TFJS_OUT / "class_names.json", "w") as f:
    json.dump(cls, f, indent=2)
print(f"Copied class_names.json -> model/class_names.json")

# ── Step 6: List output files ─────────────────────────────────────────────────
print(f"\nFiles in model/ folder:")
for fp in sorted(TFJS_OUT.iterdir()):
    print(f"  {fp.name}  ({fp.stat().st_size // 1024} KB)")

print("\n" + "=" * 60)
if success:
    print("  Export complete. Next: python serve.py")
print("=" * 60)
