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
let lastKnownPose = new Map<string, [number, number, number][]>()
let lastKnownLH = new Map<string, [number, number, number][]>()
let lastKnownRH = new Map<string, [number, number, number][]>()
let missingPoseCounts = new Map<string, number>()
let missingLHCounts = new Map<string, number>()
let missingRHCounts = new Map<string, number>()

function extractHolisticFeatures(result: any, pId: string): number[] {
  // --- Pose ---
  const poseLms = result.poseLandmarks?.[0]
  let poseMissingCount = missingPoseCounts.get(pId) || 0

  if (poseLms && poseLms.length > 0) {
    const arr = poseLms.map((lm: any) => [lm.x, lm.y, lm.z] as [number, number, number])
    while (arr.length < 33) arr.push([0, 0, 0])
    lastKnownPose.set(pId, arr)
    poseMissingCount = 0
  } else {
    poseMissingCount++
  }
  missingPoseCounts.set(pId, poseMissingCount)

  // If pose is missing and we have no history, or it's been missing for >= 5 frames
  if (poseMissingCount >= 5 || !lastKnownPose.has(pId)) {
    return new Array(FEATURES_PER_FRAME).fill(1.0)
  }
  const poseArr = lastKnownPose.get(pId)!

  // --- Left Hand ---
  const lhLms = result.leftHandLandmarks?.[0]
  let lhMissingCount = missingLHCounts.get(pId) || 0

  if (lhLms && lhLms.length > 0) {
    const arr = lhLms.map((lm: any) => [lm.x, lm.y, lm.z] as [number, number, number])
    while (arr.length < 21) arr.push([0, 0, 0])
    lastKnownLH.set(pId, arr)
    lhMissingCount = 0
  } else {
    lhMissingCount++
  }
  missingLHCounts.set(pId, lhMissingCount)

  // --- Right Hand ---
  const rhLms = result.rightHandLandmarks?.[0]
  let rhMissingCount = missingRHCounts.get(pId) || 0

  if (rhLms && rhLms.length > 0) {
    const arr = rhLms.map((lm: any) => [lm.x, lm.y, lm.z] as [number, number, number])
    while (arr.length < 21) arr.push([0, 0, 0])
    lastKnownRH.set(pId, arr)
    rhMissingCount = 0
  } else {
    rhMissingCount++
  }
  missingRHCounts.set(pId, rhMissingCount)

  // Calculate shoulder parameters from poseArr
  const ls = poseArr[11] || [0, 0, 0]
  const rs = poseArr[12] || [0, 0, 0]
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

  // Normalization helper
  const normalize = ([x, y, z]: [number, number, number]) => [
    (x - mid[0]) / shoulderWidth,
    (y - mid[1]) / shoulderWidth,
    (z - mid[2]) / shoulderWidth
  ]

  // Normalize pose features
  const normalizedPose = poseArr.flatMap(normalize)

  // Normalize hands. If missing >= 5 frames or no history, use 1.0
  const normalizedLH = (lhMissingCount >= 5 || !lastKnownLH.has(pId)) 
    ? new Array(21 * 3).fill(1.0) 
    : lastKnownLH.get(pId)!.flatMap(normalize)

  const normalizedRH = (rhMissingCount >= 5 || !lastKnownRH.has(pId)) 
    ? new Array(21 * 3).fill(1.0) 
    : lastKnownRH.get(pId)!.flatMap(normalize)

  return [...normalizedPose, ...normalizedLH, ...normalizedRH]
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
  const activeFrames = buffer.filter(frame => frame.some(val => val !== 0 && val !== 1.0)).length
  if (activeFrames < 6) {
    return null
  }

  let finalBuffer: number[][] = []
  if (buffer.length < WINDOW_SIZE) {
    finalBuffer = [...buffer]
    const padCount = WINDOW_SIZE - buffer.length
    for (let i = 0; i < padCount; i++) {
      finalBuffer.push(new Array(FEATURES_PER_FRAME).fill(1.0))
    }
  } else {
    // Sliding window logic: simply take the last 30 frames
    finalBuffer = buffer.slice(-WINDOW_SIZE)
  }

  try {
    const predictions = await runInference(finalBuffer)

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
      const pId = message.participantId || "default"
      const now = Date.now()
      const lastUpdate = lastBufferUpdate.get(pId) || 0

      // 1. CHECK TIME FIRST to save CPU/GPU overhead
      if (now - lastUpdate < 60) {
        sendResponse({ status: "skipped_frame" })
        return
      }
      
      // Update timer
      lastBufferUpdate.set(pId, now)

      // 2. DO THE HEAVY LIFTING ONLY IF FRAME IS NOT SKIPPED
      const { dataUrl } = message
      const img = new Image()
      await new Promise((resolve, reject) => {
        img.onload = resolve
        img.onerror = () => reject(new Error("Failed to decode image data URL"))
        img.src = dataUrl
      })

      // VIDEO mode requires strictly increasing timestamps
      videoTimestamp += 66
      const result = holisticLandmarker!.detectForVideo(img, videoTimestamp)

      const features = extractHolisticFeatures(result, pId)

      let responseGesture: string | null = null
      let responseTop5: any[] | null = null

      // Hands detected = either left or right hand landmarks present (access [0] for nested [][])
      const handsDetected = (result.leftHandLandmarks?.[0] && result.leftHandLandmarks[0].length > 0) ||
        (result.rightHandLandmarks?.[0] && result.rightHandLandmarks[0].length > 0)

      if (!frameBuffers.has(pId)) frameBuffers.set(pId, [])
      const buffer = frameBuffers.get(pId)!
      const emptyCount = emptyFrameCounts.get(pId) || 0

      if (handsDetected) {
        buffer.push(features)
        emptyFrameCounts.set(pId, 0)

        if (buffer.length > 45) buffer.shift()

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

          if (buffer.length > 45) buffer.shift()

          if (newEmptyCount >= 5) {
            const prediction = await triggerSegmentationInference(buffer, pId)
            if (prediction) {
              responseGesture = prediction.gesture
              responseTop5 = prediction.top5
            }
            // Wipe out the history on a long pause
            buffer.length = 0
            emptyFrameCounts.set(pId, 0)
            lastKnownPose.delete(pId)
            lastKnownLH.delete(pId)
            lastKnownRH.delete(pId)
            missingPoseCounts.set(pId, 0)
            missingLHCounts.set(pId, 0)
            missingRHCounts.set(pId, 0)
          }
        }
      }

      const bufferRef = frameBuffers.get(pId) || []
      const response: any = {
        handsDetected,
        bufferProgress: bufferRef.length,
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
