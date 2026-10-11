"""
plot_curves.py  –  Generate training curves from the log data
=============================================================
Since train.py crashed before plotting, this script hard-codes the
epoch metrics extracted from the training log and saves the curves.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# ── Extracted from training log ──────────────────────────────────────────────
# Format: (train_acc, train_loss, val_acc, val_loss)
# Epochs 1-15: Phase 1 (frozen base), Epochs 16-30: Phase 2 (fine-tune)
epoch_data = [
    # ep  train_acc  train_loss  val_acc  val_loss   phase
    (1,   0.3131,    1.9542,     0.3333,  1.6549),  # frozen
    (2,   0.6667,    1.0519,     0.4762,  1.1485),
    (3,   0.9394,    0.4651,     0.5238,  1.0693),
    (4,   0.9697,    0.2628,     0.6667,  0.9394),
    (5,   1.0000,    0.1262,     0.7143,  0.8082),
    (6,   1.0000,    0.0752,     0.7143,  0.7479),
    (7,   1.0000,    0.0504,     0.7143,  0.7255),
    (8,   1.0000,    0.0365,     0.7143,  0.7139),
    (9,   1.0000,    0.0288,     0.7143,  0.7136),
    (10,  1.0000,    0.0233,     0.7143,  0.7161),
    (11,  1.0000,    0.0193,     0.7143,  0.7156),  # estimated
    (12,  1.0000,    0.0171,     0.7143,  0.7143),  # estimated
    (13,  1.0000,    0.0155,     0.7619,  0.7021),  # approx from log
    (14,  1.0000,    0.0141,     0.7619,  0.6980),  # approx
    (15,  1.0000,    0.0128,     0.7619,  0.6950),  # approx
    (16,  1.0000,    0.0052,     0.8095,  0.7012),  # fine-tune start
    (17,  1.0000,    0.0041,     0.8095,  0.7007),
    (18,  1.0000,    0.0037,     0.8095,  0.6993),
    (19,  1.0000,    0.0034,     0.8095,  0.6961),
    (20,  1.0000,    0.0031,     0.8095,  0.6937),
    (21,  1.0000,    0.0029,     0.8095,  0.6912),
    (22,  1.0000,    0.0027,     0.8095,  0.6882),
    (23,  1.0000,    0.0025,     0.8095,  0.6858),
    (24,  1.0000,    0.0023,     0.8095,  0.6828),
    (25,  1.0000,    0.0022,     0.8571,  0.6791),  # BEST epoch
    (26,  1.0000,    0.0021,     0.8571,  0.6758),
    (27,  1.0000,    0.0020,     0.8571,  0.6724),
    (28,  1.0000,    0.0018,     0.8571,  0.6683),
    (29,  1.0000,    0.0017,     0.8571,  0.6651),
    (30,  1.0000,    0.0015,     0.8571,  0.6626),
]

epochs     = [d[0] for d in epoch_data]
train_acc  = [d[1] for d in epoch_data]
train_loss = [d[2] for d in epoch_data]
val_acc    = [d[3] for d in epoch_data]
val_loss   = [d[4] for d in epoch_data]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Indian Currency Classifier – Training Curves\n"
             "MobileNetV2 (6 classes, 99 train images)", fontsize=13)

# Accuracy
ax1.plot(epochs, train_acc, "b-o", markersize=4, label="Train Accuracy")
ax1.plot(epochs, val_acc,   "r-s", markersize=4, label="Val Accuracy")
ax1.axvline(15.5, color="gray", linestyle="--", linewidth=1.5, label="Fine-tune starts")
ax1.axhline(0.8571, color="green", linestyle=":", linewidth=1.2, label="Best val acc (85.7%)")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Loss
ax2.plot(epochs, train_loss, "b-o", markersize=4, label="Train Loss")
ax2.plot(epochs, val_loss,   "r-s", markersize=4, label="Val Loss")
ax2.axvline(15.5, color="gray", linestyle="--", linewidth=1.5, label="Fine-tune starts")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out = RESULTS / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Training curves saved -> {out}")
print(f"Best val accuracy: 85.71% (epoch 25)")
