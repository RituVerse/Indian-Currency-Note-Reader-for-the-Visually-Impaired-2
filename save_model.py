"""
save_model.py  –  Export best_model.keras → SavedModel + training curves
=========================================================================
Run this after train.py completes (even if train.py crashed at the save step).
It loads the best checkpoint and exports it in all needed formats.
"""
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tensorflow import keras

SAVE_DIR = Path("model_saved")
RESULTS  = Path("results")
RESULTS.mkdir(exist_ok=True)

keras_path = SAVE_DIR / "best_model.keras"
if not keras_path.exists():
    raise FileNotFoundError(f"Cannot find {keras_path}. Did train.py complete?")

print(f"Loading best model from {keras_path}…")
model = keras.models.load_model(str(keras_path))

# Export as TF SavedModel (required for tensorflowjs_converter)
saved_path = str(SAVE_DIR / "saved_model")
print(f"Exporting SavedModel to {saved_path}…")
model.export(saved_path)
print("Done. Saved model is ready for TF.js export.")
print("Next: python export_tfjs.py")
