"""
export_tfjs_graph.py  –  Convert SavedModel → TF.js graph model
================================================================
Strategy: mock the broken imports (distutils, tensorflow_hub) BEFORE
importing tensorflowjs, then call convert_tf_saved_model() directly.
This is the graph model converter — it does NOT check Keras version.
"""

import sys, types, os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── Mock distutils (removed in Python 3.12+) ──────────────────────────────────
fake_distutils = types.ModuleType("distutils")
fake_version   = types.ModuleType("distutils.version")
class LooseVersion:
    def __init__(self, v="0"): self.vstring = str(v)
    def __le__(self,o): return True
    def __lt__(self,o): return True
    def __ge__(self,o): return True
    def __gt__(self,o): return True
    def __eq__(self,o): return False
    def __repr__(self): return self.vstring
fake_version.LooseVersion   = LooseVersion
fake_version.StrictVersion  = LooseVersion
fake_distutils.version      = fake_version
sys.modules["distutils"]         = fake_distutils
sys.modules["distutils.version"] = fake_version

# ── Mock tensorflow_hub (only needed for hub.KerasLayer models — not ours) ───
fake_hub = types.ModuleType("tensorflow_hub")
fake_hub.load       = lambda *a, **kw: None
fake_hub.KerasLayer = object
fake_hub.resolve    = lambda *a, **kw: None
sys.modules["tensorflow_hub"] = fake_hub

# ── Now import tensorflowjs internals ─────────────────────────────────────────
print("Importing tensorflowjs converter …")
from tensorflowjs.converters import tf_saved_model_conversion_v2

from pathlib import Path
SAVE_DIR = Path("model_saved")
TFJS_OUT = Path("model")
TFJS_OUT.mkdir(exist_ok=True)

saved_model_path = str(SAVE_DIR / "saved_model")
output_path      = str(TFJS_OUT)

print(f"Converting {saved_model_path} -> {output_path} …")
print("(This may take 1-2 minutes)\n")

try:
    tf_saved_model_conversion_v2.convert_tf_saved_model(
        saved_model_dir          = saved_model_path,
        output_dir               = output_path,
        signature_def            = "serving_default",
        saved_model_tags         = "serve",
        quantization_dtype_map   = None,
        skip_op_check            = True,
        strip_debug_ops          = True,
        weight_shard_size_bytes  = 4 * 1024 * 1024,  # 4 MB shards
    )
    print("\nConversion SUCCEEDED!")
    success = True
except Exception as e:
    print(f"\nConversion failed: {e}")
    import traceback; traceback.print_exc()
    success = False

if success:
    import json
    # Copy class names
    with open("class_names.json") as f:
        cls = json.load(f)
    with open(TFJS_OUT / "class_names.json", "w") as f:
        json.dump(cls, f, indent=2)

    print("\nFiles in model/:")
    for fp in sorted(TFJS_OUT.iterdir()):
        print(f"  {fp.name}  ({fp.stat().st_size // 1024} KB)")
    print("\nNext: python serve.py  then open http://localhost:8000")
else:
    print("""
The graph model converter also failed.
The manual layers-model export (model/model.json + group1-shard1of1.bin)
is still available. Update script.js to use tf.loadLayersModel() instead
of tf.loadGraphModel() to load it.
""")
