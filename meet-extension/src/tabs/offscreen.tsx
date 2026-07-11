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

    const modelUrl = chrome.runtime.getURL("assets/arabic_sign_model_large.onnx")
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

function normalizeLandmarks(landmarks: any[]): number[] {
  const wrist = landmarks[0]
  const centered = landmarks.map(lm => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: lm.z - wrist.z
  }))

  const distances = centered.map(c => Math.sqrt(c.x * c.x + c.y * c.y + c.z * c.z))
  const maxDist = Math.max(...distances)

  if (maxDist > 0) {
    return centered.flatMap(c => [c.x / maxDist, c.y / maxDist, c.z / maxDist])
  } else {
    return centered.flatMap(c => [c.x, c.y, c.z])
  }
}

function extractHands(result: any): number[] {
  let lh = new Array(63).fill(0)
  let rh = new Array(63).fill(0)
  // Use hand_landmarks (image-space [0,1]) and apply the same normalization
  // (centering around the wrist and scaling) as the training pipeline.
  if (result.landmarks && result.landmarks.length > 0) {
    for (let i = 0; i < result.handedness.length; i++) {
      const label = result.handedness[i][0].categoryName
      const normalized = normalizeLandmarks(result.landmarks[i])
      if (label === "Left") lh = normalized
      if (label === "Right") rh = normalized
    }
  }
  return [...lh, ...rh]
}

async function runInference(buffer: number[][]): Promise<{ word: string, prob: number }[]> {
  if (!ort || !ortSession || labels.length === 0) return []
  try {
    const flatBuffer = new Float32Array(WINDOW_SIZE * FEATURES_PER_FRAME)
    for (let i = 0; i < WINDOW_SIZE; i++) flatBuffer.set(buffer[i], i * FEATURES_PER_FRAME)
    const tensor = new ort.Tensor("float32", flatBuffer, [1, WINDOW_SIZE, FEATURES_PER_FRAME])
    const results = await ortSession.run({ [ortSession.inputNames[0]]: tensor })
    // The ONNX model already outputs softmax probabilities because of the final Dense layer
    // activation="softmax" in the Keras model. We do NOT need to apply softmax again.
    // Applying softmax twice squashes the probabilities to ~1/502 (0.2%).
    const probs = results[ortSession.outputNames[0]].data as Float32Array

    const top5 = Array.from(probs)
      .map((prob, idx) => ({ word: labels[idx] || "", prob }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5)

    return top5
  } catch (e: any) {
    console.error("[ASL] Inference error:", e)
    throw new Error("Inference failed: " + e.message)
  }
}

let emptyFrameCounts = new Map<string, number>()

async function triggerSegmentationInference(buffer: number[][], pId: string): Promise<{ gesture: string, top5: any[] } | null> {
  // Activity check: Ensure the buffer contains at least 6 frames with detected hands
  // to avoid running inference on rest positions or noise
  const activeFrames = buffer.filter(frame => frame.some(val => val !== 0)).length
  if (activeFrames < 6) {
    buffer.length = 0 // Clear buffer in-place
    emptyFrameCounts.set(pId, 0)
    return null
  }

  // Prepare exactly 30 frames
  let finalBuffer: number[][] = []
  if (buffer.length < WINDOW_SIZE) {
    // Pad end with zeros (matches the zero-mask training pipeline)
    finalBuffer = [...buffer]
    const padCount = WINDOW_SIZE - buffer.length
    for (let i = 0; i < padCount; i++) {
      finalBuffer.push(new Array(FEATURES_PER_FRAME).fill(0))
    }
  } else if (buffer.length > WINDOW_SIZE) {
    // Resample to exactly 30 frames using linear spacing
    for (let i = 0; i < WINDOW_SIZE; i++) {
      const idx = Math.floor((i * buffer.length) / WINDOW_SIZE)
      finalBuffer.push(buffer[idx])
    }
  } else {
    finalBuffer = buffer
  }

  try {
    const predictions = await runInference(finalBuffer)
    buffer.length = 0 // Clear buffer in-place
    emptyFrameCounts.set(pId, 0)

    if (predictions && predictions.length > 0) {
      const topPrediction = predictions[0]
      if (topPrediction.prob > 0.15) {
        if (topPrediction.word !== lastPredictions.get(pId)) {
          lastPredictions.set(pId, topPrediction.word)
          return { gesture: topPrediction.word, top5: predictions }
        }
      } else {
        lastPredictions.delete(pId)
      }
    }
  } catch (err) {
    console.error("[ASL] Inference error in segmentation:", err)
    buffer.length = 0
    emptyFrameCounts.set(pId, 0)
  }
  return null
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
      // We only process if at least 60ms has passed since the last push.
      let shouldPush = false
      if (now - lastUpdate >= 60) {
        shouldPush = true
        lastBufferUpdate.set(pId, now)
      }

      let responseGesture: string | null = null
      let responseTop5: any[] | null = null

      if (shouldPush) {
        const handsDetected = result.landmarks && result.landmarks.length > 0

        if (!frameBuffers.has(pId)) frameBuffers.set(pId, [])
        const buffer = frameBuffers.get(pId)!
        const emptyCount = emptyFrameCounts.get(pId) || 0

        if (handsDetected) {
          buffer.push(features)
          emptyFrameCounts.set(pId, 0)
          
          // Force trigger if sequence is too long to prevent lagging
          if (buffer.length >= 45) {
            const prediction = await triggerSegmentationInference(buffer, pId)
            if (prediction) {
              responseGesture = prediction.gesture
              responseTop5 = prediction.top5
            }
          }
        } else {
          // No hands detected
          if (buffer.length > 0) {
            const newEmptyCount = emptyCount + 1
            emptyFrameCounts.set(pId, newEmptyCount)
            
            // Push the empty frame
            buffer.push(features)

            if (newEmptyCount >= 5) {
              // User paused signing: process the accumulated sequence
              const prediction = await triggerSegmentationInference(buffer, pId)
              if (prediction) {
                responseGesture = prediction.gesture
                responseTop5 = prediction.top5
              }
            }
          }
        }
      }

      const buffer = frameBuffers.get(pId) || []
      const response: any = {
        handsDetected: result.landmarks && result.landmarks.length > 0,
        bufferProgress: buffer.length,
        bufferSize: WINDOW_SIZE,
      }
      const isLocal = process.env.PLASMO_PUBLIC_SHOW_LANDMARKS === "true" || process.env.NODE_ENV === "development"
      if (isLocal) {
        response.landmarks = result.landmarks || []
      }
      if (responseGesture && responseTop5) {
        response.gesture = responseGesture
        response.top5 = responseTop5
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
