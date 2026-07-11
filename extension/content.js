/**
 * extension/content.js
 * ====================
 * Injected into Google Meet pages.
 *
 * Responsibilities
 * ----------------
 * 1. Detect participant <video> elements.
 * 2. Capture each video frame via an off-screen <canvas>.
 * 3. Extract MediaPipe hand landmarks (via importScripts CDN build).
 * 4. Send landmark arrays to the backend via WebSocket.
 * 5. Receive predictions and overlay RTL Arabic subtitles.
 * 6. Display FPS counter and confidence indicator.
 */

"use strict";

// ─────────────────────────────────────────────────────────────────────────────
// Constants / Settings (overridden from popup via chrome.storage.sync)
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULTS = {
  enabled:        true,
  backendUrl:     "ws://localhost:8000/ws/",
  targetFps:      15,
  showFps:        true,
  showConfidence: true,
  fontSize:       "22px",
  opacity:        0.85,
};

let SETTINGS = { ...DEFAULTS };

// ─────────────────────────────────────────────────────────────────────────────
// State
// ─────────────────────────────────────────────────────────────────────────────

/** Map<videoElement, ParticipantSession> */
const sessions = new Map();

let handsModel = null;   // MediaPipe Hands instance (shared)

// ─────────────────────────────────────────────────────────────────────────────
// Load settings from chrome.storage
// ─────────────────────────────────────────────────────────────────────────────

chrome.storage.sync.get(DEFAULTS, (stored) => {
  SETTINGS = { ...DEFAULTS, ...stored };
  if (SETTINGS.enabled) init();
});

chrome.storage.onChanged.addListener((changes) => {
  for (const [key, { newValue }] of Object.entries(changes)) {
    SETTINGS[key] = newValue;
  }
  if (!SETTINGS.enabled) teardownAll();
  else if (sessions.size === 0) init();
});

// ─────────────────────────────────────────────────────────────────────────────
// MediaPipe Hands initialisation
// ─────────────────────────────────────────────────────────────────────────────

async function loadMediaPipeHands() {
  if (handsModel) return handsModel;

  const { Hands } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/hands.js"
  );

  const hands = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
  });

  hands.setOptions({
    maxNumHands:          2,
    modelComplexity:      1,
    minDetectionConfidence:  0.6,
    minTrackingConfidence:   0.5,
  });

  await hands.initialize();
  handsModel = hands;
  return hands;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-participant session
// ─────────────────────────────────────────────────────────────────────────────

class ParticipantSession {
  /**
   * @param {HTMLVideoElement} videoEl
   */
  constructor(videoEl) {
    this.videoEl    = videoEl;
    this.clientId   = crypto.randomUUID();
    this.canvas     = document.createElement("canvas");
    this.ctx        = this.canvas.getContext("2d");
    this.ws         = null;
    this.overlay    = null;
    this.fpsEl      = null;
    this.confEl     = null;
    this.lastFrameTs = 0;
    this.frameCount  = 0;
    this.fps         = 0;
    this.active      = false;

    this._createOverlay();
    this._connectWebSocket();
    this._startLoop();
  }

  // ── Overlay ──────────────────────────────────────────────────────────────

  _createOverlay() {
    const container = this.videoEl.parentElement || document.body;

    // Subtitle bar
    this.overlay = document.createElement("div");
    Object.assign(this.overlay.style, {
      position:        "absolute",
      bottom:          "8px",
      left:            "50%",
      transform:       "translateX(-50%)",
      maxWidth:        "90%",
      background:      `rgba(0,0,0,${SETTINGS.opacity})`,
      color:           "#fff",
      fontSize:        SETTINGS.fontSize,
      fontFamily:      "'Noto Sans Arabic', 'Arial', sans-serif",
      direction:       "rtl",
      textAlign:       "right",
      padding:         "6px 14px",
      borderRadius:    "8px",
      zIndex:          "9999",
      pointerEvents:   "none",
      display:         "none",
      transition:      "opacity 0.3s",
    });
    container.style.position = "relative";
    container.appendChild(this.overlay);

    // FPS badge
    if (SETTINGS.showFps) {
      this.fpsEl = document.createElement("div");
      Object.assign(this.fpsEl.style, {
        position:  "absolute",
        top:       "4px",
        left:      "4px",
        fontSize:  "10px",
        color:     "#0f0",
        zIndex:    "9999",
        pointerEvents: "none",
      });
      container.appendChild(this.fpsEl);
    }

    // Confidence badge
    if (SETTINGS.showConfidence) {
      this.confEl = document.createElement("div");
      Object.assign(this.confEl.style, {
        position:  "absolute",
        top:       "4px",
        right:     "4px",
        fontSize:  "10px",
        color:     "#ff0",
        zIndex:    "9999",
        pointerEvents: "none",
      });
      container.appendChild(this.confEl);
    }
  }

  // ── WebSocket ─────────────────────────────────────────────────────────────

  _connectWebSocket() {
    const url = SETTINGS.backendUrl + this.clientId;
    this.ws = new WebSocket(url);

    this.ws.onopen    = () => console.log("[ArSL] WS connected:", this.clientId);
    this.ws.onclose   = () => setTimeout(() => this._connectWebSocket(), 2000);
    this.ws.onerror   = (e) => console.warn("[ArSL] WS error", e);
    this.ws.onmessage = (ev) => this._handleMessage(JSON.parse(ev.data));
  }

  _handleMessage(msg) {
    // Update subtitle
    if (msg.sentence_completed) {
      this._showSubtitle(msg.sentence_completed);
    } else if (msg.buffer_preview) {
      this._showSubtitle(msg.buffer_preview, /* final= */ false);
    }

    // Confidence badge
    if (this.confEl && msg.raw_conf !== undefined) {
      const pct = Math.round(msg.raw_conf * 100);
      this.confEl.textContent = `conf: ${pct}%`;
    }
  }

  _showSubtitle(text, final = true) {
    this.overlay.textContent = text;
    this.overlay.style.display = "block";
    this.overlay.style.opacity = "1";

    if (final) {
      clearTimeout(this._hideTimer);
      this._hideTimer = setTimeout(() => {
        this.overlay.style.opacity = "0";
        setTimeout(() => (this.overlay.style.display = "none"), 300);
      }, 4000);
    }
  }

  // ── Frame loop ────────────────────────────────────────────────────────────

  async _startLoop() {
    this.active = true;
    const hands = await loadMediaPipeHands();
    const intervalMs = 1000 / SETTINGS.targetFps;

    hands.onResults((results) => this._onHandResults(results));

    const tick = async () => {
      if (!this.active) return;

      const now = performance.now();
      if (now - this.lastFrameTs < intervalMs) {
        requestAnimationFrame(tick);
        return;
      }

      this.lastFrameTs = now;
      this.frameCount++;

      // FPS calculation
      if (this.frameCount % 15 === 0 && this.fpsEl) {
        this.fps = Math.round(15000 / (now - (this._fpsTs || now)));
        this._fpsTs = now;
        this.fpsEl.textContent = `FPS: ${this.fps}`;
      }

      // Capture frame
      const video = this.videoEl;
      if (video.readyState >= 2 && video.videoWidth > 0) {
        this.canvas.width  = video.videoWidth;
        this.canvas.height = video.videoHeight;
        this.ctx.drawImage(video, 0, 0);
        await hands.send({ image: this.canvas });
      }

      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
  }

  _onHandResults(results) {
    if (!results.multiHandLandmarks || results.multiHandLandmarks.length === 0)
      return;

    // Flatten all landmarks from both hands into one array
    const landmarks = [];
    for (const hand of results.multiHandLandmarks) {
      for (const lm of hand) {
        landmarks.push(lm.x, lm.y, lm.z);
      }
    }

    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        landmarks,
        timestamp: performance.now(),
      }));
    }
  }

  // ── Teardown ──────────────────────────────────────────────────────────────

  destroy() {
    this.active = false;
    if (this.ws) this.ws.close();
    if (this.overlay)  this.overlay.remove();
    if (this.fpsEl)    this.fpsEl.remove();
    if (this.confEl)   this.confEl.remove();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Video element discovery (MutationObserver)
// ─────────────────────────────────────────────────────────────────────────────

function isGoogleMeetVideo(el) {
  return (
    el.tagName === "VIDEO" &&
    el.srcObject instanceof MediaStream &&
    el.videoWidth > 0
  );
}

function attachToVideo(video) {
  if (sessions.has(video)) return;
  console.log("[ArSL] Attaching to video element", video);
  sessions.set(video, new ParticipantSession(video));
}

function init() {
  // Attach to any already-present videos
  document.querySelectorAll("video").forEach((v) => {
    if (isGoogleMeetVideo(v)) attachToVideo(v);
  });

  // Watch for dynamically added videos
  const observer = new MutationObserver((mutations) => {
    for (const { addedNodes } of mutations) {
      addedNodes.forEach((node) => {
        if (node.nodeType !== Node.ELEMENT_NODE) return;
        if (isGoogleMeetVideo(node)) attachToVideo(node);
        node.querySelectorAll?.("video").forEach((v) => {
          if (isGoogleMeetVideo(v)) attachToVideo(v);
        });
      });
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });

  // Also handle video load events (src not yet ready on append)
  document.addEventListener("loadedmetadata", (e) => {
    if (e.target.tagName === "VIDEO" && isGoogleMeetVideo(e.target)) {
      attachToVideo(e.target);
    }
  }, true);
}

function teardownAll() {
  sessions.forEach((session) => session.destroy());
  sessions.clear();
}
