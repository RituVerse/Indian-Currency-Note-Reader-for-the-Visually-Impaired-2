# Indian Currency Note Identifier
**Entry:** BUILD WITH ML — Coding Connoisseurs, FoET, University of Lucknow

---

## Project Status

| Step | Status |
|------|--------|
| 1 – Dataset inspection & duplicate removal | ✅ Done |
| 2 – Data split (70/15/15, seed 42) | ⏳ In progress |
| 3 – Training (frozen base + fine-tune) | ⬜ Remaining |
| 4 – Evaluation & confusion matrix | ⬜ Remaining |
| 5 – Export to TensorFlow.js | ⬜ Remaining |
| 6 – Web app (HTML/CSS/JS) | ⬜ Remaining |
| 7 – Local server test | ⬜ Remaining |
| 8 – Final README | ⬜ Remaining |

### Decisions
- Classes: 10, 20, 50, 100, 200, 500, no_note. **2000-rupee class dropped** (withdrawn 2023).
- no_note class needs phone photos (not in the Kaggle dataset) — see "Phone Photos" section below.
- Model: MobileNetV2 (ImageNet weights, 224×224). Transfer learning: frozen base → fine-tune last 20 layers.
- Split: 70/15/15, random seed 42. Duplicates (MD5) removed before splitting.
- Augmentation: rotation, brightness, zoom. **No horizontal/vertical flips** (they would flip the note's printing).
- Confidence threshold: 85%. Below → "Not sure. Please show the note again."
- Training settings: batch 8, ~20 fine-tune layers (tuned for low-RAM / CPU machines).
- **No voice output / speechSynthesis** — visual result only.

### Known Issues / Errors
- Python 3.14.4 — TF 2.22.0rc0 (only rc build available for this Python version). If it crashes, downgrade Python to 3.11 in a venv.
- `pip` needs `--trusted-host` flags (proxy on this network).
- TensorFlowJS converter may conflict with TF+Keras 3; fallback is `tf_keras` route or SavedModel export.

### Files
| File | Purpose |
|------|---------|
| `.gitignore` | Keeps dataset and binaries out of git |
| `inspect_data.py` | Step 1: count images, find duplicates, report corrupt files |
| `split_dataset.py` | Step 2: deduplicate + split archive into dataset_split/ |
| `train.py` | Step 3: MobileNetV2 transfer learning + fine-tuning |
| `evaluate.py` | Step 4: test-set accuracy, confusion matrix |
| `export_tfjs.py` | Step 5: convert SavedModel → TF.js |
| `model/` | TF.js weight files (committed to git) |
| `results/` | Training curves + confusion matrix images |
| `index.html` | Web app entry point |
| `style.css` | Accessible high-contrast styles |
| `script.js` | TF.js inference, webcam, upload logic |
| `serve.py` | Tiny local HTTP server for testing |

### How to Reproduce (from a clean clone)
```bash
# 1. Put the "archive" folder back in this directory
# 2. Create & activate venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# 3. Run each step
python inspect_data.py
python split_dataset.py      # writes dataset_split/
python train.py              # writes model_saved/ + results/
python evaluate.py           # writes results/confusion_matrix.png
python export_tfjs.py        # writes model/ folder
# 4. Test the web app
python serve.py              # open http://localhost:8000
```

### Phone Photos Needed
The `no_note` class must come from your own phone photos.
Expected folder: `my_photos/no_note/` (hand, table, empty frame — at least 20 images).
Also recommended: add extra photos of each note class for better real-world accuracy.

---

## Problem Statement
India uses banknotes in denominations of ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500.
These notes differ in colour and size, but many users — the elderly, people with low vision,
tourists, and anyone unfamiliar with the designs — struggle to tell them apart quickly.

## Proposed Solution
A browser-based app that uses the device's camera or a photo upload to classify the
denomination in real time using a MobileNetV2 neural network exported to TensorFlow.js.
The answer appears in large, high-contrast text (e.g. **"This is a ₹500 note"**),
with a confidence bar. No installation needed — it runs entirely in the browser.

## Target Users
- Elderly people and people with low vision
- Tourists unfamiliar with Indian currency
- Anyone needing quick denomination confirmation

## Dataset
**Kaggle dataset:** "Indian Currency Dataset" — *paste author name and licence from the Kaggle page here.*
- 7 original classes (Train + Test folders already split by Kaggle)
- ~22 images per class in Train, ~6 per class in Test
- 15 exact duplicates removed (MD5 hash comparison)
- 2000-rupee class excluded (withdrawn from circulation May 2023)

**Own photos:** Photographs taken by Ritesh Yadav.

## Training Process
1. All Kaggle images (excluding 2000-rupee) are combined and re-split 70/15/15 (train/val/test), random seed 42, after removing MD5 duplicates.
2. Phone photos are added to the training and validation sets. A separate, never-trained-on subset is kept as the real-world test set.
3. MobileNetV2 (pretrained on ImageNet) is used as the base. The top layers are replaced by: GlobalAveragePooling → Dense(128, ReLU) → Dropout(0.3) → Dense(7, softmax).
4. **Phase 1 – Frozen training:** base layers frozen, train top layers for 15 epochs.
5. **Phase 2 – Fine-tuning:** last 20 layers of MobileNetV2 unfrozen, trained for 15 more epochs at a lower learning rate (1e-5).
6. Augmentation applied: random rotation (±15°), brightness (±20%), zoom (±10%). No flips.

## Test Results
*(To be filled in after training — see results/ folder)*

| Set | Accuracy |
|-----|----------|
| Kaggle test set (small, noisy) | — |
| Real-world phone-photo test set | — |

> **Note:** The Kaggle test set has only ~3–4 images per class after deduplication.
> Test accuracy on this set is **noisy** and should be read as counts (e.g. 3/4 correct = 75%).
> The phone-photo test set is the honest measure of real-world performance.

## Per-Class Results
*(To be filled in — see results/confusion_matrix.png)*

## Implementation
- **Model:** MobileNetV2 (TensorFlow/Keras), exported to TF.js format
- **Web app:** Vanilla HTML + CSS + JavaScript, TensorFlow.js v4
- **Camera:** `getUserMedia` API (webcam) + drag-and-drop / file upload
- **Inference:** Runs fully in the browser — no server, no data sent anywhere
- **Hosting:** Static files on Vercel (or any static host)

## Limitations
- Small dataset (~22 images/class from Kaggle) — model may struggle with unusual lighting or angles
- Folded, torn, or heavily worn notes may be misclassified
- Old note designs vs. new Mahatma Gandhi Series 2 designs may confuse the model
- No ₹2000 note (officially withdrawn)
- Real-world accuracy depends heavily on phone/webcam quality and lighting

## Future Work
- Voice/audio output (speech synthesis) for visually impaired users
- Larger, more diverse dataset
- On-device model (TFLite) for a native Android/iOS app
- Support for detecting multiple notes in one frame

## Credits
- **Dataset:** Indian Currency Dataset on Kaggle — *paste author + licence*
- **Own photos:** Ritesh Yadav
- **AI Assistants:** Google Antigravity (Claude Sonnet 4.6 Thinking) — used for code generation, project structuring, and debugging guidance
- **Framework:** TensorFlow / Keras, TensorFlow.js, MobileNetV2 (Google)
