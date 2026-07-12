import { FilesetResolver, HolisticLandmarker } from "@mediapipe/tasks-vision"

// We load it globally via a <script> tag in offscreen.html.
declare global {
  interface Window {
    ort: any;
  }
}
let ort: any = null
let holisticLandmarker: HolisticLandmarker | null = null
let ortSession: any = null
let labels: string[] = []
let initError: string | null = null

// Holistic: pose(33×3=99) + left_hand(21×3=63) + right_hand(21×3=63) = 225
const WINDOW_SIZE = 30
const FEATURES_PER_FRAME = 225
let frameBuffers = new Map<string, number[][]>()
let lastPredictions = new Map<string, string>()
let videoTimestamp = 0
let lastBufferUpdate = new Map<string, number>()

async function init() {
  console.log("[ASL] Initializing HolisticLandmarker...")
  try {
    const vision = await FilesetResolver.forVisionTasks(
      chrome.runtime.getURL("assets/tasks-vision-wasm")
    )
    holisticLandmarker = await HolisticLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: chrome.runtime.getURL("assets/holistic_landmarker.task")
      },
      runningMode: "VIDEO",
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence: 0.5,
      minHandLandmarksConfidence: 0.5
    })
    console.log("[ASL] HolisticLandmarker ready.")
  } catch (e: any) {
    console.error("[ASL] HolisticLandmarker init failed:", e)
    initError = "HolisticLandmarker: " + e.message
    return
  }

  console.log("[ASL] Loading ONNX Runtime...")
  try {
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script")
      script.src = chrome.runtime.getURL("assets/ort-wasm/ort.all.min.js")
      script.onload = () => resolve()
      script.onerror = () => reject(new Error("Failed to load ONNX script"))
      document.head.appendChild(script)
    })

    ort = window.ort
    if (!ort) throw new Error("window.ort is undefined after script load")

    ort.env.wasm.numThreads = 1
    ort.env.wasm.wasmPaths = chrome.runtime.getURL("assets/ort-wasm/")

    const modelUrl = chrome.runtime.getURL("assets/arabic_sign_model_medium_new.onnx")
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

/**
 * Extracts and normalizes holistic features to EXACTLY match the training pipeline:
 *
 *   [pose(33×3=99), left_hand(21×3=63), right_hand(21×3=63)] = 225
 *
 * Normalization (from interpolate_and_normalize in training):
 *   - Global anchor: mid-shoulder = (pose[11] + pose[12]) / 2
 *   - Global scale:  shoulder_width = ||pose[11] - pose[12]||
 *   - Formula:       (landmark - mid_shoulder) / shoulder_width
 *
 * Missing landmarks are zero-padded (zeros after normalization = at origin).
 */
function extractHolisticFeatures(result: any): number[] {
  // In MediaPipe 0.10.x, all landmark fields are NormalizedLandmark[][]
  // (array of detections). We take [0] for the first (and only) detected person/hand.

  // --- Pose landmarks (33 × 3) ---
  let poseArr: [number, number, number][] = Array.from({ length: 33 }, () => [0, 0, 0])
  const poseLms = result.poseLandmarks?.[0]   // first detected pose
  if (poseLms && poseLms.length > 0) {
    poseArr = poseLms.map((lm: any) => [lm.x, lm.y, lm.z] as [number, number, number])
  }

  // --- Left hand (21 × 3) — direct from holistic, not by handedness label ---
  let lhArr: [number, number, number][] = Array.from({ length: 21 }, () => [0, 0, 0])
  const lhLms = result.leftHandLandmarks?.[0]  // first detected left hand
  if (lhLms && lhLms.length > 0) {
    lhArr = lhLms.map((lm: any) => [lm.x, lm.y, lm.z] as [number, number, number])
  }

  // --- Right hand (21 × 3) ---
  let rhArr: [number, number, number][] = Array.from({ length: 21 }, () => [0, 0, 0])
  const rhLms = result.rightHandLandmarks?.[0] // first detected right hand
  if (rhLms && rhLms.length > 0) {
    rhArr = rhLms.map((lm: any) => [lm.x, lm.y, lm.z] as [number, number, number])
  }
  // Ensure poseArr always has exactly 33 elements (pad with zeros if partial)
  while (poseArr.length < 33) poseArr.push([0, 0, 0])

  // --- Shoulder normalization (matches training's interpolate_and_normalize) ---
  const ls = poseArr[11] ?? [0, 0, 0]  // left shoulder  (safe fallback)
  const rs = poseArr[12] ?? [0, 0, 0]  // right shoulder (safe fallback)
  const mid = [
    (ls[0] + rs[0]) / 2,
    (ls[1] + rs[1]) / 2,
    (ls[2] + rs[2]) / 2
  ]
  const shoulderWidth = Math.sqrt(
    Math.pow(ls[0] - rs[0], 2) +
    Math.pow(ls[1] - rs[1], 2) +
    Math.pow(ls[2] - rs[2], 2)
  ) || 1e-6

  const normalize = ([x, y, z]: [number, number, number]) => [
    (x - mid[0]) / shoulderWidth,
    (y - mid[1]) / shoulderWidth,
    (z - mid[2]) / shoulderWidth
  ]

  const allLandmarks = [...poseArr, ...lhArr, ...rhArr]
  return allLandmarks.flatMap(normalize)
}

async function runInference(buffer: number[][]): Promise<{ word: string, prob: number }[]> {
  if (!ort || !ortSession || labels.length === 0) return []
  try {
    const flatBuffer = new Float32Array(WINDOW_SIZE * FEATURES_PER_FRAME)
    for (let i = 0; i < WINDOW_SIZE; i++) flatBuffer.set(buffer[i], i * FEATURES_PER_FRAME)
    const tensor = new ort.Tensor("float32", flatBuffer, [1, WINDOW_SIZE, FEATURES_PER_FRAME])
    const results = await ortSession.run({ [ortSession.inputNames[0]]: tensor })
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
  const activeFrames = buffer.filter(frame => frame.some(val => val !== 0)).length
  if (activeFrames < 6) {
    buffer.length = 0
    emptyFrameCounts.set(pId, 0)
    return null
  }

  let finalBuffer: number[][] = []
  if (buffer.length < WINDOW_SIZE) {
    finalBuffer = [...buffer]
    const padCount = WINDOW_SIZE - buffer.length
    for (let i = 0; i < padCount; i++) {
      finalBuffer.push(new Array(FEATURES_PER_FRAME).fill(0))
    }
  } else if (buffer.length > WINDOW_SIZE) {
    for (let i = 0; i < WINDOW_SIZE; i++) {
      const idx = Math.floor((i * buffer.length) / WINDOW_SIZE)
      finalBuffer.push(buffer[idx])
    }
  } else {
    finalBuffer = buffer
  }

  try {
    const predictions = await runInference(finalBuffer)
    buffer.length = 0
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

  if (!holisticLandmarker || !ortSession) {
    sendResponse({ status: "loading" })
    return false
  }

  ; (async () => {
    try {
      const { dataUrl } = message
      const pId = message.participantId || "default"

      const img = new Image()
      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = () => reject(new Error("Failed to decode image data URL"))
        img.src = dataUrl
      })

      // VIDEO mode requires strictly increasing timestamps
      videoTimestamp += 33
      const result = holisticLandmarker!.detectForVideo(img, videoTimestamp)

      const features = extractHolisticFeatures(result)
      const now = Date.now()
      const lastUpdate = lastBufferUpdate.get(pId) || 0

      // ~15 FPS to match training (every-other-frame at 30fps = 66ms)
      let shouldPush = false
      if (now - lastUpdate >= 60) {
        shouldPush = true
        lastBufferUpdate.set(pId, now)
      }

      let responseGesture: string | null = null
      let responseTop5: any[] | null = null

      // Hands detected = either left or right hand landmarks present (access [0] for nested [][])
      const handsDetected = (result.leftHandLandmarks?.[0] && result.leftHandLandmarks[0].length > 0) ||
        (result.rightHandLandmarks?.[0] && result.rightHandLandmarks[0].length > 0)

      if (shouldPush) {
        if (!frameBuffers.has(pId)) frameBuffers.set(pId, [])
        const buffer = frameBuffers.get(pId)!
        const emptyCount = emptyFrameCounts.get(pId) || 0

        if (handsDetected) {
          buffer.push(features)
          emptyFrameCounts.set(pId, 0)

          if (buffer.length >= 45) {
            const prediction = await triggerSegmentationInference(buffer, pId)
            if (prediction) {
              responseGesture = prediction.gesture
              responseTop5 = prediction.top5
            }
          }
        } else {
          if (buffer.length > 0) {
            const newEmptyCount = emptyCount + 1
            emptyFrameCounts.set(pId, newEmptyCount)
            buffer.push(features)

            if (newEmptyCount >= 5) {
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
        handsDetected,
        bufferProgress: buffer.length,
        bufferSize: WINDOW_SIZE,
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

export default function OffscreenPage() {
  return null
}
