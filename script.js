/**
 * script.js  —  Indian Currency Identifier  (Interactive Redesign)
 * ═══════════════════════════════════════════════════════════════════
 * Features:
 *  · Load TF.js graph model from model/
 *  · Live webcam with front/back camera flip
 *  · Drag-and-drop / click image upload
 *  · Anti-flicker: update display only when stable for ~2 seconds
 *  · Per-class mini probability bars
 *  · Last-5-results history strip
 *  · No voice / speech output anywhere
 */
"use strict";

// ── Constants ─────────────────────────────────────────────────────────────────
const MODEL_URL          = "./model/model.json";
const THRESHOLD          = 0.85;   // confidence cutoff
const PREDICT_MS         = 200;    // how often to run inference (ms)
const STABLE_FRAMES      = 10;     // frames before updating display (~2s)
const MAX_HISTORY        = 5;      // how many recent results to show

// Denomination display labels
const LABELS = {
  "10": "₹10", "20": "₹20", "50": "₹50",
  "100": "₹100", "200": "₹200", "500": "₹500",
  "no_note": "No note"
};
// Icons per class
const ICONS = {
  "10":"💵","20":"💵","50":"💵","100":"💵","200":"💵","500":"💵",
  "no_note":"🔲"
};

// ── State ─────────────────────────────────────────────────────────────────────
let model      = null;
let classNames = [];
let stream     = null;
let facingMode = "environment";   // back camera default
let predTimer  = null;
let stableCount = 0;
let lastClass   = null;
let displayedClass = null;
let history    = [];   // [{label, conf, icon}]

// ── DOM refs ──────────────────────────────────────────────────────────────────
const webcamEl       = document.getElementById("webcam");
const snapshotCanvas = document.getElementById("snapshot");
const modelPill      = document.getElementById("modelPill");
const pillText       = document.getElementById("pillText");
const scanOverlay    = document.getElementById("scanOverlay");
const videoPlaceholder = document.getElementById("videoPlaceholder");
const resultCard     = document.getElementById("resultCard");
const resultIcon     = document.getElementById("resultIcon");
const resultLabel    = document.getElementById("resultLabel");
const resultText     = document.getElementById("resultText");
const confCard       = document.getElementById("confCard");
const confBar        = document.getElementById("confBar");
const confPct        = document.getElementById("confPct");
const miniProbs      = document.getElementById("miniProbs");
const historyCard    = document.getElementById("historyCard");
const historyList    = document.getElementById("historyList");
const startBtn       = document.getElementById("startBtn");
const stopBtn        = document.getElementById("stopBtn");
const flipBtn        = document.getElementById("flipBtn");
const dropZone       = document.getElementById("dropZone");
const dropContent    = document.getElementById("dropContent");
const uploadedImg    = document.getElementById("uploadedImg");

// ── 1. Boot ───────────────────────────────────────────────────────────────────
loadModel();

// ── 2. Model loading ──────────────────────────────────────────────────────────
async function loadModel() {
  setPill("loading", "Loading AI model…");
  try {
    const resp = await fetch("./model/class_names.json");
    if (!resp.ok) throw new Error("class_names.json not found");
    classNames = await resp.json();

    model = await tf.loadGraphModel(MODEL_URL);
    setPill("ready", `Model ready · ${classNames.length} classes`);
    buildMiniProbRows();   // create the mini bar DOM once
    console.log("Model loaded. Classes:", classNames);
  } catch (err) {
    setPill("error", "Model failed to load");
    console.error(err);
  }
}

function setPill(state, text) {
  modelPill.className = "model-pill " + state;
  pillText.textContent = text;
}

// ── 3. Tab switching ──────────────────────────────────────────────────────────
function switchTab(tab) {
  document.getElementById("cameraPane").style.display = tab === "camera" ? "" : "none";
  document.getElementById("uploadPane").style.display = tab === "upload" ? "" : "none";
  document.getElementById("tabCamera").classList.toggle("active", tab === "camera");
  document.getElementById("tabUpload").classList.toggle("active", tab === "upload");

  if (tab === "upload" && stream) stopCamera();
}

// ── 4. Camera ─────────────────────────────────────────────────────────────────
async function startCamera() {
  if (!model) { alert("Model not loaded yet. Please wait a moment."); return; }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode }, width: 640, height: 480 },
      audio: false
    });
    webcamEl.srcObject = stream;
    await webcamEl.play();

    videoPlaceholder.style.display = "none";
    scanOverlay.style.display      = "";
    startBtn.disabled = true;
    stopBtn.disabled  = false;
    flipBtn.disabled  = false;

    predTimer = setInterval(predictFromWebcam, PREDICT_MS);
    setResult("waiting", "📷", "LIVE", "Scanning for a banknote…");
  } catch (err) {
    alert("Camera error: " + err.message +
          "\n\nMake sure you opened this via http://localhost (not file://) and granted camera permission.");
  }
}

function stopCamera() {
  clearInterval(predTimer); predTimer = null;
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  webcamEl.srcObject = null;
  videoPlaceholder.style.display = "";
  scanOverlay.style.display      = "none";
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  flipBtn.disabled  = true;
  resetState();
}

async function flipCamera() {
  facingMode = (facingMode === "environment") ? "user" : "environment";
  stopCamera();
  await startCamera();
}

// ── 5. Webcam predict loop ────────────────────────────────────────────────────
async function predictFromWebcam() {
  if (!webcamEl.videoWidth) return;
  const ctx = snapshotCanvas.getContext("2d");
  snapshotCanvas.width  = webcamEl.videoWidth;
  snapshotCanvas.height = webcamEl.videoHeight;
  ctx.drawImage(webcamEl, 0, 0);

  const { topClass, topConf, probs } = await runModel(snapshotCanvas);

  // Anti-flicker: only update after STABLE_FRAMES consecutive same prediction
  if (topClass === lastClass) {
    stableCount++;
  } else {
    stableCount = 1;
    lastClass   = topClass;
  }

  if (stableCount >= STABLE_FRAMES && topClass !== displayedClass) {
    displayedClass = topClass;
    updateDisplay(topClass, topConf, probs, true);
  } else if (stableCount >= STABLE_FRAMES) {
    updateConfBar(topConf, probs);
  }
}

// ── 6. Upload ─────────────────────────────────────────────────────────────────
function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) processFile(file);
}

function handleDragOver(e) {
  e.preventDefault();
  dropZone.classList.add("drag-over");
}

function handleDrop(e) {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith("image/")) processFile(file);
}

function processFile(file) {
  if (!model) { alert("Model not loaded yet."); return; }
  const url = URL.createObjectURL(file);
  uploadedImg.src = url;
  uploadedImg.style.display = "";
  dropContent.style.display = "none";
  setResult("waiting", "🔍", "ANALYSING", "Processing image…");

  uploadedImg.onload = async () => {
    const { topClass, topConf, probs } = await runModel(uploadedImg);
    updateDisplay(topClass, topConf, probs, false);
    URL.revokeObjectURL(url);
  };
}

function clearUpload() {
  uploadedImg.style.display    = "none";
  dropContent.style.display    = "";
  uploadedImg.src              = "";
  document.getElementById("fileInput").value = "";
  resetState();
}

// ── 7. Core inference ─────────────────────────────────────────────────────────
async function runModel(source) {
  return tf.tidy(() => {
    let t = tf.browser.fromPixels(source);
    t = tf.image.resizeBilinear(t, [224, 224]);
    t = t.toFloat().div(127.5).sub(1.0);   // MobileNetV2 preprocessing: [0,255]→[-1,1]
    t = t.expandDims(0);
    const out   = model.predict(t);
    const probs = Array.from(out.dataSync());
    const topIdx  = probs.indexOf(Math.max(...probs));
    return { topClass: classNames[topIdx], topConf: probs[topIdx], probs };
  });
}

// ── 8. Display ────────────────────────────────────────────────────────────────
function updateDisplay(cls, conf, probs, addToHistory) {
  confCard.style.display = "";
  updateConfBar(conf, probs);

  if (cls === "no_note") {
    setResult("nonote", "🔲", "NO NOTE", "No banknote detected");
  } else if (conf < THRESHOLD) {
    setResult("unsure", "🤔", "NOT SURE",
      `Best guess: ${LABELS[cls] || cls}\nTry again with better lighting.`);
  } else {
    setResult("detected", ICONS[cls] || "💵", "IDENTIFIED",
      `This is a\n<span class="denom-big">${LABELS[cls] || cls} note</span>`);
  }

  if (addToHistory && cls !== "no_note" && conf >= THRESHOLD) {
    addHistory(cls, conf);
  }
}

function setResult(state, icon, label, html) {
  resultCard.className = "result-card state-" + state;
  resultIcon.textContent  = icon;
  resultLabel.textContent = label;
  resultText.innerHTML    = html.replace(/\n/g, "<br>");
}

function updateConfBar(conf, probs) {
  const pct = Math.round(conf * 100);
  confPct.textContent = pct + "%";
  confBar.style.width = pct + "%";
  confBar.className   = "bar-fill " + (pct >= 85 ? "high" : pct >= 50 ? "medium" : "low");
  confBar.parentElement.setAttribute("aria-valuenow", pct);
  updateMiniProbs(probs);
}

// ── 9. Mini probability bars ──────────────────────────────────────────────────
function buildMiniProbRows() {
  miniProbs.innerHTML = "";
  classNames.forEach(cls => {
    const label = LABELS[cls] || cls;
    // Colour cycles through accent colours per class
    const colour = cls === "no_note" ? "#6366f1" : "#f5a623";
    miniProbs.insertAdjacentHTML("beforeend", `
      <div class="mini-row" id="mini-${cls}">
        <span class="mini-label">${label}</span>
        <div class="mini-track">
          <div class="mini-bar" id="mbar-${cls}"
               style="width:0%;background:${colour}"></div>
        </div>
        <span class="mini-pct" id="mpct-${cls}">0%</span>
      </div>`);
  });
}

function updateMiniProbs(probs) {
  const topIdx = probs.indexOf(Math.max(...probs));
  classNames.forEach((cls, i) => {
    const pct = Math.round(probs[i] * 100);
    const bar = document.getElementById("mbar-" + cls);
    const pctEl = document.getElementById("mpct-" + cls);
    const row   = document.getElementById("mini-" + cls);
    if (!bar) return;
    bar.style.width     = pct + "%";
    pctEl.textContent   = pct + "%";
    row.className       = "mini-row" + (i === topIdx ? " top" : "");
  });
}

// ── 10. Result history ────────────────────────────────────────────────────────
function addHistory(cls, conf) {
  history.unshift({ label: LABELS[cls] || cls, conf: Math.round(conf * 100), icon: ICONS[cls] || "💵" });
  if (history.length > MAX_HISTORY) history.pop();
  renderHistory();
}

function renderHistory() {
  if (history.length === 0) { historyCard.style.display = "none"; return; }
  historyCard.style.display = "";
  historyList.innerHTML = history.map(h =>
    `<div class="history-item">
       <span class="hi-label">${h.icon} ${h.label}</span>
       <span class="hi-conf">${h.conf}% confident</span>
     </div>`
  ).join("");
}

// ── 11. Helpers ───────────────────────────────────────────────────────────────
function resetState() {
  stableCount = 0; lastClass = null; displayedClass = null;
  setResult("waiting", "🔍", "WAITING", "Point your camera at a banknote, or upload a photo.");
  confCard.style.display = "none";
}
