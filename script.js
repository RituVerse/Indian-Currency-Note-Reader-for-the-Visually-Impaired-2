/**
 * script.js  –  Indian Currency Note Identifier
 * ═══════════════════════════════════════════════
 * This file handles everything the page does:
 *   1. Load the TensorFlow.js model from the model/ folder.
 *   2. Stream the webcam into the <video> element.
 *   3. Every 200ms: grab a frame, run it through the neural network,
 *      and update the result text — BUT only if the prediction changed
 *      or has been stable for 2 seconds (prevents flickering).
 *   4. Also handle uploaded images.
 *
 * There is NO voice / speech output anywhere in this file.
 *
 * Terminology quick-guide:
 *   tensor     = a multi-dimensional array that TF.js uses for numbers
 *   softmax    = the output layer that converts raw scores to probabilities
 *   confidence = the probability (0–1) that the model gives to its top guess
 */

"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────

// Path to the TF.js model (relative to index.html)
// The converter produces either a graph model or a layers model depending on
// which export method succeeded.  We try graph model first, then layers model.
const MODEL_URL = "./model/model.json";

// Confidence below this → "Not sure" message
const THRESHOLD = 0.85;

// How often (milliseconds) to grab a frame from the webcam and predict
const PREDICT_INTERVAL_MS = 200;

// Stable-result logic: if the SAME class is predicted for this many
// consecutive frames, we update the display.  200ms × 10 = 2 seconds.
const STABLE_FRAMES = 10;

// ── DOM element references ────────────────────────────────────────────────────
const webcamEl        = document.getElementById("webcam");
const snapshotCanvas  = document.getElementById("snapshot");
const startBtn        = document.getElementById("startBtn");
const stopBtn         = document.getElementById("stopBtn");
const fileInput       = document.getElementById("fileInput");
const uploadPreview   = document.getElementById("uploadPreview");
const uploadedImg     = document.getElementById("uploadedImg");
const resultBox       = document.getElementById("resultBox");
const resultText      = document.getElementById("resultText");
const confidenceWrap  = document.getElementById("confidenceWrapper");
const confidenceBar   = document.getElementById("confidenceBar");
const confidencePct   = document.getElementById("confidencePct");
const allProbsEl      = document.getElementById("allProbs");
const modelStatusEl   = document.getElementById("modelStatus");

// ── App state ─────────────────────────────────────────────────────────────────
let model      = null;    // the loaded TF.js model
let classNames = [];      // ["10", "20", "50", "100", "200", "500", "no_note"]
let stream     = null;    // the webcam MediaStream
let predTimer  = null;    // setInterval handle for the prediction loop
let stableCount = 0;      // how many consecutive frames gave the same class
let lastClass   = null;   // the class predicted in the previous frame
let displayedClass = null;  // what is currently shown on screen

// ── 1. Load model on page start ──────────────────────────────────────────────

async function loadModel() {
  setStatus("⏳ Loading AI model…");
  try {
    // Fetch class names (the order must match the model's output layer)
    const resp = await fetch("./model/class_names.json");
    if (!resp.ok) throw new Error("Could not load class_names.json");
    classNames = await resp.json();

    // Load the TF.js model. tf.loadGraphModel reads model.json + weight files.
    model = await tf.loadGraphModel(MODEL_URL);
    setStatus(`✅ Model ready (${classNames.length} classes)`);
    console.log("Model loaded. Classes:", classNames);
  } catch (err) {
    setStatus(`❌ Error loading model: ${err.message}`);
    console.error(err);
  }
}

// ── 2. Webcam handling ────────────────────────────────────────────────────────

startBtn.addEventListener("click", async () => {
  if (!model) { alert("Model not loaded yet. Please wait."); return; }

  try {
    // Ask the browser for camera access.
    // facingMode: "environment" = back camera on phones (better for notes)
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: "environment" }, width: 640, height: 480 },
      audio: false,
    });
    webcamEl.srcObject = stream;
    await webcamEl.play();

    startBtn.disabled = true;
    stopBtn.disabled  = false;

    // Start the prediction loop
    predTimer = setInterval(predictFromWebcam, PREDICT_INTERVAL_MS);
    showResult("waiting", "📷 Camera active. Point at a banknote…", null);
  } catch (err) {
    alert("Could not access camera:\n" + err.message +
          "\n\nTip: make sure you are using http://localhost and have given camera permission.");
    console.error(err);
  }
});

stopBtn.addEventListener("click", () => {
  clearInterval(predTimer);
  predTimer = null;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  webcamEl.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  stableCount  = 0;
  lastClass    = null;
  displayedClass = null;
  showResult("waiting", "Waiting for camera or photo…", null);
  confidenceWrap.style.display = "none";
});

// ── 3. Webcam prediction loop ────────────────────────────────────────────────

async function predictFromWebcam() {
  if (!webcamEl.videoWidth) return;  // video not ready yet

  // Draw the current video frame onto the hidden canvas
  const ctx = snapshotCanvas.getContext("2d");
  snapshotCanvas.width  = webcamEl.videoWidth;
  snapshotCanvas.height = webcamEl.videoHeight;
  ctx.drawImage(webcamEl, 0, 0);

  // Run the neural network on this canvas frame
  const { topClass, topConf, probs } = await runModel(snapshotCanvas);

  // ── Stable-result anti-flicker logic ────────────────────────────────────
  // We don't update the display on EVERY frame — only when the model has
  // been consistently predicting the SAME class for ~2 seconds.
  if (topClass === lastClass) {
    stableCount++;
  } else {
    stableCount = 1;
    lastClass   = topClass;
  }

  // Update display only when stable AND (class changed OR forced update)
  if (stableCount >= STABLE_FRAMES && topClass !== displayedClass) {
    displayedClass = topClass;
    updateDisplay(topClass, topConf, probs);
  } else if (stableCount >= STABLE_FRAMES) {
    // Same class, just update the confidence bar (no text flicker)
    updateConfidenceBar(topConf, probs);
  }
}

// ── 4. Image upload handling ──────────────────────────────────────────────────

fileInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  if (!model) { alert("Model not loaded yet."); return; }

  // Stop webcam if it was running
  if (stream) stopBtn.click();

  // Show the uploaded image as a preview
  const objectUrl = URL.createObjectURL(file);
  uploadedImg.src = objectUrl;
  uploadPreview.style.display = "block";

  // Wait for the image to finish loading, then run the model
  uploadedImg.onload = async () => {
    showResult("waiting", "🔍 Analysing…", null);
    const { topClass, topConf, probs } = await runModel(uploadedImg);
    updateDisplay(topClass, topConf, probs);
    URL.revokeObjectURL(objectUrl);  // free memory
  };
});

// Also allow clicking the upload-zone div (for keyboard users)
document.querySelector(".upload-zone").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") fileInput.click();
});

// ── 5. Core inference function ────────────────────────────────────────────────

/**
 * runModel(imageSource)
 * Takes an HTMLImageElement, HTMLCanvasElement, or HTMLVideoElement,
 * preprocesses it, runs the TF.js model, and returns the top prediction.
 *
 * @returns {topClass: string, topConf: number, probs: number[]}
 */
async function runModel(imageSource) {
  // tf.tidy() automatically cleans up intermediate tensors (frees memory).
  // This is important in a prediction loop — without it, memory would grow.
  return tf.tidy(() => {
    // Step A: Convert the image to a tensor (a 3D array: height × width × 3 channels)
    let tensor = tf.browser.fromPixels(imageSource);

    // Step B: Resize to 224×224 (MobileNetV2's expected input size)
    tensor = tf.image.resizeBilinear(tensor, [224, 224]);

    // Step C: MobileNetV2 was trained with pixel values in [-1, 1], not [0, 255].
    // Formula: (pixel / 127.5) - 1
    tensor = tensor.toFloat().div(127.5).sub(1.0);

    // Step D: Add a batch dimension → shape becomes [1, 224, 224, 3]
    tensor = tensor.expandDims(0);

    // Step E: Run the model! It returns an array of probabilities (softmax output).
    const predictions = model.predict(tensor);

    // Step F: Convert tensor to a plain JavaScript array
    const probs = Array.from(predictions.dataSync());

    // Step G: Find the class with the highest probability
    const topIdx  = probs.indexOf(Math.max(...probs));
    const topConf = probs[topIdx];
    const topClass = classNames[topIdx];

    return { topClass, topConf, probs };
  });
}

// ── 6. Display update functions ────────────────────────────────────────────────

/**
 * updateDisplay: called when we have a stable new prediction.
 * Updates the big result text AND the confidence bar.
 */
function updateDisplay(cls, conf, probs) {
  confidenceWrap.style.display = "block";

  if (cls === "no_note") {
    showResult("nonote", "🔲 No note detected", conf);
  } else if (conf < THRESHOLD) {
    // Below the 85% confidence threshold → don't commit to an answer
    showResult("unsure",
      `Not sure. Please show the note again.\n(Best guess: ₹${cls})`,
      conf
    );
  } else {
    // Confident prediction — the main use case
    showResult("detected",
      `This is a\n<span class="denom-highlight">₹${formatDenom(cls)} note</span>`,
      conf
    );
  }

  updateConfidenceBar(conf, probs);
}

/**
 * showResult: change the result box's appearance and text.
 * state = "waiting" | "detected" | "unsure" | "nonote"
 */
function showResult(state, html, conf) {
  // Remove all state classes then add the new one
  resultBox.className = "result-box result-" + state;

  // Set the inner HTML (we use innerHTML to allow the <span> inside)
  resultText.innerHTML = html.replace(/\n/g, "<br>");
}

/**
 * updateConfidenceBar: update just the bar and percentages without
 * touching the big result text.
 */
function updateConfidenceBar(conf, probs) {
  const pct = Math.round(conf * 100);
  confidencePct.textContent = pct + "%";

  // Move the bar fill
  confidenceBar.style.width = pct + "%";

  // Colour: green ≥ 85%, amber ≥ 50%, red < 50%
  confidenceBar.className = "bar-fill " +
    (pct >= 85 ? "high" : pct >= 50 ? "medium" : "low");

  // Update the aria value for screen readers
  confidenceBar.parentElement.setAttribute("aria-valuenow", pct);

  // Show a small breakdown of all class probabilities
  allProbsEl.innerHTML = classNames
    .map((cls, i) => {
      const p = Math.round(probs[i] * 100);
      const label = cls === "no_note" ? "no_note" : `₹${formatDenom(cls)}`;
      return `<span>${label}: ${p}%</span>`;
    })
    .join("  ·  ");
}

/** Format denomination: "100" → "100", "2000" → "2,000" etc. */
function formatDenom(cls) {
  const n = parseInt(cls, 10);
  if (isNaN(n)) return cls;
  return n >= 1000 ? n.toLocaleString("en-IN") : String(n);
}

/** Update the small status line at the bottom of the result section. */
function setStatus(msg) {
  modelStatusEl.textContent = msg;
}

// ── 7. Boot the app ────────────────────────────────────────────────────────────
loadModel();
