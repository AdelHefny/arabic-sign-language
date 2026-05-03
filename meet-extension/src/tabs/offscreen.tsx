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
const FEATURES_PER_FRAME = 126
let frameBuffers = new Map<string, number[][]>()

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
      runningMode: "IMAGE",
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

    const modelUrl = chrome.runtime.getURL("assets/arabic_sign_video_model_small.onnx")
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

async function runInference(buffer: number[][]): Promise<string> {
  if (!ort || !ortSession || labels.length === 0) return ""
  try {
    const flatBuffer = new Float32Array(WINDOW_SIZE * FEATURES_PER_FRAME)
    for (let i = 0; i < WINDOW_SIZE; i++) flatBuffer.set(buffer[i], i * FEATURES_PER_FRAME)
    const tensor = new ort.Tensor("float32", flatBuffer, [1, WINDOW_SIZE, FEATURES_PER_FRAME])
    const results = await ortSession.run({ [ortSession.inputNames[0]]: tensor })
    const logits = results[ortSession.outputNames[0]].data as Float32Array
    let maxIdx = 0
    for (let i = 1; i < logits.length; i++) if (logits[i] > logits[maxIdx]) maxIdx = i
    return labels[maxIdx] || ""
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
      const { data, width, height } = message.imageData
      const pId = message.participantId || "default"
      
      const result = handLandmarker.detect(new ImageData(new Uint8ClampedArray(data), width, height))
      const features = extractHands(result)
      
      if (!frameBuffers.has(pId)) frameBuffers.set(pId, [])
      const buffer = frameBuffers.get(pId)!
      
      buffer.push(features)
      if (buffer.length > WINDOW_SIZE) buffer.shift()

      const response: any = {
        landmarks: result.landmarks || [],
        handsDetected: result.landmarks && result.landmarks.length > 0,
        bufferProgress: buffer.length,
        bufferSize: WINDOW_SIZE,
      }

      if (buffer.length === WINDOW_SIZE) {
        try {
          const prediction = await runInference(buffer)
          if (prediction) response.gesture = prediction
        } catch (infErr: any) {
          response.error = infErr.message
        }
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
