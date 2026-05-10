import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision"

// We load it globally via a <script> tag in offscreen.html.
declare global {
  interface Window {
    ort: any;
  }
}
let ort: any = null
let handLandmarker: HandLandmarker | null = null
let ortSession: any = null
let labels: string[] = []
let initError: string | null = null

const WINDOW_SIZE = 30
const STRIDE = 10
const FEATURES_PER_FRAME = 126
let frameBuffers = new Map<string, number[][]>()
let lastPredictions = new Map<string, string>()
// Monotonically increasing timestamp for VIDEO mode (must never go backwards)
let videoTimestamp = 0
let lastBufferUpdate = new Map<string, number>()

async function init() {
  console.log("[ASL] Initializing MediaPipe...")
  try {
    const vision = await FilesetResolver.forVisionTasks(
      chrome.runtime.getURL("assets/tasks-vision-wasm")
    )
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: chrome.runtime.getURL("assets/hand_landmarker.task")
      },
      runningMode: "VIDEO",
      numHands: 2
    })
    console.log("[ASL] MediaPipe ready.")
  } catch (e: any) {
    console.error("[ASL] MediaPipe init failed:", e)
    initError = "MediaPipe: " + e.message
    return
  }

  console.log("[ASL] Loading ONNX Runtime...")
  try {
    // Inject the script tag dynamically to bypass Plasmo's bundler resolving the path
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script")
      script.src = chrome.runtime.getURL("assets/ort-wasm/ort.all.min.js")
      script.onload = () => resolve()
      script.onerror = () => reject(new Error("Failed to load ONNX script"))
      document.head.appendChild(script)
    })

    ort = window.ort
    if (!ort) throw new Error("window.ort is undefined after script load")

    // Disable threading to avoid SharedArrayBuffer requirement in extension context
    ort.env.wasm.numThreads = 1
    ort.env.wasm.wasmPaths = chrome.runtime.getURL("assets/ort-wasm/")

    const modelUrl = chrome.runtime.getURL("assets/arabic_sign_model_medium.onnx")
    ortSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ["wasm"] })

    const labelsUrl = chrome.runtime.getURL("assets/labels.json")
    labels = await (await fetch(labelsUrl)).json()
    console.log(`[ASL] Model ready. ${labels.length} labels loaded.`)
  } catch (e: any) {
    console.error("[ASL] ONNX model load failed:", e)
    initError = "ONNX: " + e.message
    return
  }
}

init()

function extractHands(result: any): number[] {
  let lh = new Array(63).fill(0)
  let rh = new Array(63).fill(0)
  // Use hand_landmarks (image-space [0,1]) — this matches the training pipeline
  // which used result.hand_landmarks, NOT world_landmarks.
  if (result.landmarks && result.landmarks.length > 0) {
    for (let i = 0; i < result.handedness.length; i++) {
      const label = result.handedness[i][0].categoryName
      const flat = result.landmarks[i].flatMap((lm: any) => [lm.x, lm.y, lm.z])
      if (label === "Left") lh = flat
      if (label === "Right") rh = flat
    }
  }
  return [...lh, ...rh]
}

async function runInference(buffer: number[][]): Promise<{ word: string, prob: number } | null> {
  if (!ort || !ortSession || labels.length === 0) return null
  try {
    const flatBuffer = new Float32Array(WINDOW_SIZE * FEATURES_PER_FRAME)
    for (let i = 0; i < WINDOW_SIZE; i++) flatBuffer.set(buffer[i], i * FEATURES_PER_FRAME)
    const tensor = new ort.Tensor("float32", flatBuffer, [1, WINDOW_SIZE, FEATURES_PER_FRAME])
    const results = await ortSession.run({ [ortSession.inputNames[0]]: tensor })
    // The ONNX model already outputs softmax probabilities because of the final Dense layer
    // activation="softmax" in the Keras model. We do NOT need to apply softmax again.
    // Applying softmax twice squashes the probabilities to ~1/502 (0.2%).
    const probs = results[ortSession.outputNames[0]].data as Float32Array

    let maxIdx = 0
    let maxProb = 0
    for (let i = 0; i < probs.length; i++) {
      if (probs[i] > maxProb) {
        maxProb = probs[i]
        maxIdx = i
      }
    }

    return { word: labels[maxIdx] || "", prob: maxProb }
  } catch (e: any) {
    console.error("[ASL] Inference error:", e)
    throw new Error("Inference failed: " + e.message)
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.target !== "offscreen" || message.type !== "PROCESS_FRAME") return false

  if (initError) {
    sendResponse({ error: initError })
    return false
  }

  if (!handLandmarker || !ortSession) {
    sendResponse({ status: "loading" })
    return false
  }

  ; (async () => {
    try {
      const { dataUrl, width, height } = message
      const pId = message.participantId || "default"

      // Decode the dataUrl to an Image object for MediaPipe
      const img = new Image()
      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = (e) => reject(new Error("Failed to decode image data URL"))
        img.src = dataUrl
      })

      // VIDEO mode requires strictly increasing timestamps.
      // Increment by 33ms (~30fps) for smooth tracking
      videoTimestamp += 33
      const result = handLandmarker!.detectForVideo(img, videoTimestamp)

      const features = extractHands(result)
      const now = Date.now()
      const lastUpdate = lastBufferUpdate.get(pId) || 0

      // The ASL model expects 15fps (66ms interval). 
      // We only push to the buffer if at least 60ms has passed since the last push.
      // This allows us to track hands at 30fps for the UI while keeping the model input correct.
      if (now - lastUpdate >= 60) {
        if (!frameBuffers.has(pId)) frameBuffers.set(pId, [])
        const buffer = frameBuffers.get(pId)!
        buffer.push(features)
        lastBufferUpdate.set(pId, now)
      }

      const buffer = frameBuffers.get(pId) || []
      const response: any = {
        landmarks: result.landmarks || [],
        handsDetected: result.landmarks && result.landmarks.length > 0,
        bufferProgress: buffer.length,
        bufferSize: WINDOW_SIZE,
      }

      if (buffer.length === WINDOW_SIZE) {
        try {
          const prediction = await runInference(buffer)
          if (prediction) {
            // The model was trained with CategoricalCrossentropy(label_smoothing=0.1) across 502 classes.
            // This pulls the maximum theoretical confidence down to ~0.90.
            // In a real-time sliding window context with partial gestures, a confidence of 15-25% 
            // is extremely strong (random chance is ~0.19%).
            if (prediction.prob > 0.15) {
              if (prediction.word !== lastPredictions.get(pId)) {
                response.gesture = prediction.word
                lastPredictions.set(pId, prediction.word)
              }
            } else {
              lastPredictions.delete(pId)
            }
          }
        } catch (infErr: any) {
          response.error = infErr.message
        }
        buffer.splice(0, STRIDE)
      }

      sendResponse(response)
    } catch (err) {
      console.error("[ASL] Processing error:", err)
      sendResponse({ error: "Processing error" })
    }
  })()

  return true
})

// Plasmo requires a default export for src/tabs/ pages to bundle them correctly
export default function OffscreenPage() {
  return null
}
