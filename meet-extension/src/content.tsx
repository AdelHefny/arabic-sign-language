import type { PlasmoCSConfig } from "plasmo"
import { useEffect, useState, useRef } from "react"
import { observeParticipants } from "~lib/ParticipantObserver"

export const config: PlasmoCSConfig = {
  matches: ["https://meet.google.com/*"]
}

export const getStyle = () => {
  const style = document.createElement("style")
  style.textContent = `
    #plasmo-shadow-container {
      z-index: 2147483647 !important;
      position: fixed !important;
      inset: 0 !important;
      pointer-events: none !important;
      width: 100vw !important;
      height: 100vh !important;
    }
  `
  return style
}

import { sendToBackground } from "@plasmohq/messaging"

const drawLandmarks = (canvas: HTMLCanvasElement, video: HTMLVideoElement, landmarksList: any[][]) => {
  const ctx = canvas.getContext("2d")
  if (!ctx) return

  const rect = video.getBoundingClientRect()
  const computedStyle = window.getComputedStyle(video)
  
  canvas.style.left = `${rect.left}px`
  canvas.style.top = `${rect.top}px`
  canvas.style.width = `${rect.width}px`
  canvas.style.height = `${rect.height}px`
  canvas.style.transform = computedStyle.transform
  canvas.style.transformOrigin = computedStyle.transformOrigin

  if (canvas.width !== rect.width || canvas.height !== rect.height) {
    canvas.width = rect.width
    canvas.height = rect.height
  }

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (!landmarksList || landmarksList.length === 0) return

  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20],
    [5, 9], [9, 13], [13, 17]
  ]

  landmarksList.forEach(landmarks => {
    // Draw connections
    ctx.strokeStyle = "#00ffcc"
    ctx.lineWidth = 3
    connections.forEach(([i, j]) => {
      const pt1 = landmarks[i]
      const pt2 = landmarks[j]
      if (pt1 && pt2) {
        ctx.beginPath()
        ctx.moveTo(pt1.x * canvas.width, pt1.y * canvas.height)
        ctx.lineTo(pt2.x * canvas.width, pt2.y * canvas.height)
        ctx.stroke()
      }
    })

    // Draw joints
    ctx.fillStyle = "#ff0055"
    landmarks.forEach(lm => {
      ctx.beginPath()
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 5, 0, 2 * Math.PI)
      ctx.fill()
    })
  })
}

// --- Types ---
type AppStatus = "loading" | "ready" | "error" | "no_hands" | "buffering"

// --- Status Badge shown in the top-left of each video tile ---
const StatusBadge = ({ status, detail, bufferProgress, bufferSize }: {
  status: AppStatus
  detail: string
  bufferProgress: number
  bufferSize: number
}) => {
  const configs: Record<AppStatus, { emoji: string; color: string; bg: string; label: string }> = {
    loading: { emoji: "⏳", color: "#facc15", bg: "rgba(0,0,0,0.75)", label: detail || "Loading..." },
    ready: { emoji: "🟢", color: "#4ade80", bg: "rgba(0,0,0,0.75)", label: detail || "Ready" },
    error: { emoji: "❌", color: "#f87171", bg: "rgba(80,0,0,0.85)", label: detail || "Error" },
    no_hands: { emoji: "✋", color: "#94a3b8", bg: "rgba(0,0,0,0.65)", label: "No hands detected" },
    buffering: { emoji: "📡", color: "#60a5fa", bg: "rgba(0,0,0,0.75)", label: `Building window… ${bufferProgress}/${bufferSize}` },
  }

  const cfg = configs[status]

  return (
    <div style={{
      position: "absolute",
      top: "10px",
      left: "10px",
      display: "flex",
      alignItems: "center",
      gap: "6px",
      background: cfg.bg,
      color: cfg.color,
      padding: "5px 10px",
      borderRadius: "8px",
      fontSize: "12px",
      fontFamily: "system-ui, sans-serif",
      fontWeight: 600,
      border: `1px solid ${cfg.color}44`,
      backdropFilter: "blur(4px)",
      userSelect: "none",
    }}>
      <span style={{ fontSize: "14px" }}>{cfg.emoji}</span>
      <span>ASL · {cfg.label}</span>
      {status === "buffering" && (
        <div style={{
          width: "50px",
          height: "4px",
          background: "rgba(255,255,255,0.15)",
          borderRadius: "2px",
          overflow: "hidden",
          marginLeft: "4px",
        }}>
          <div style={{
            width: `${(bufferProgress / bufferSize) * 100}%`,
            height: "100%",
            background: "#60a5fa",
            borderRadius: "2px",
            transition: "width 0.1s linear",
          }} />
        </div>
      )}
    </div>
  )
}

// --- Main video overlay per participant ---
const VideoOverlay = ({ video }: { video: HTMLVideoElement }) => {
  const captureCanvasRef = useRef<HTMLCanvasElement>(null)
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null)
  const [captions, setCaptions] = useState<string[]>([])
  const [finalSentence, setFinalSentence] = useState<string | null>(null)
  const [status, setStatus] = useState<AppStatus>("loading")
  const [statusDetail, setStatusDetail] = useState("Initializing...")
  const [bufferProgress, setBufferProgress] = useState(0)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const pauseTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const predictionSequenceRef = useRef<any[]>([])
  const BUFFER_SIZE = 30
  const participantId = useRef(Math.random().toString(36).slice(2)).current
  const isLocal = process.env.PLASMO_PUBLIC_SHOW_LANDMARKS === "true" || process.env.NODE_ENV === "development"

  const processSentence = async () => {
    if (predictionSequenceRef.current.length === 0) return
    const sequence = [...predictionSequenceRef.current]
    predictionSequenceRef.current = [] // Clear buffer
    
    try {
      const response = await sendToBackground({
        name: "process_sentence",
        body: { sequence }
      })
      if (response && response.sentence) {
        setFinalSentence(response.sentence)
        setCaptions([]) // Clear raw captions
        
        if (timeoutRef.current) clearTimeout(timeoutRef.current)
        timeoutRef.current = setTimeout(() => {
          setFinalSentence(null)
        }, 5000)
      }
    } catch (err) {
      console.error("[ASL] Error processing sentence:", err)
    }
  }

  const updatePrediction = (text: string, top5: any[]) => {
    predictionSequenceRef.current.push(top5)
    
    setCaptions(prev => {
      const newCaptions = [...prev, text]
      if (newCaptions.length > 10) return newCaptions.slice(newCaptions.length - 10)
      return newCaptions
    })
    
    if (finalSentence) setFinalSentence(null)

    if (pauseTimeoutRef.current) clearTimeout(pauseTimeoutRef.current)
    pauseTimeoutRef.current = setTimeout(() => {
      processSentence()
    }, 1500) // 1.5 seconds pause triggers generation

    if (predictionSequenceRef.current.length >= 15) {
      if (pauseTimeoutRef.current) clearTimeout(pauseTimeoutRef.current)
      processSentence()
    }
  }

  useEffect(() => {
    let active = true

    if (!captureCanvasRef.current) {
      captureCanvasRef.current = document.createElement("canvas")
    }
    const captureCanvas = captureCanvasRef.current
    const captureCtx = captureCanvas.getContext("2d", { willReadFrequently: true })!

    const processFrame = async () => {
      if (!active) return

      if (video.readyState < 2 || video.videoWidth === 0 || video.videoHeight === 0) {
        setTimeout(processFrame, 66)
        return
      }

      // Reduce resolution for performance. MediaPipe doesn't need 640x480 for detection.
      // 320x240 is 4x less data and still very accurate for landmarks.
      if (captureCanvas.width !== 320) {
        captureCanvas.width = 320
        captureCanvas.height = 240
      }

      try {
        captureCtx.drawImage(video, 0, 0, 320, 240)
        
        // Use JPEG string instead of raw array. 
        // Array.from() + JSON serialization of 1.2M elements was the bottleneck.
        // A 320x240 JPEG is ~15KB vs ~1MB for a JSON array.
        const dataUrl = captureCanvas.toDataURL("image/jpeg", 0.7)

        try {
          const response = await sendToBackground({
            name: "process_frame",
            body: {
              participantId,
              dataUrl,
              width: 320,
              height: 240
            }
          })

          if (!active) return

          if (!response) {
            setTimeout(processFrame, 100)
            return
          }

          // --- Update UI state based on response ---
          if (response.status === "loading") {
            setStatus("loading")
            setStatusDetail("Loading model...")
            setTimeout(processFrame, 200)
            return
          }

          if (response.error) {
            setStatus("error")
            setStatusDetail(response.error)
            setTimeout(processFrame, 500)
            return
          }


          const progress = response.bufferProgress ?? 0
          setBufferProgress(progress)

          if (!response.handsDetected) {
            setStatus("no_hands")
          } else if (progress < BUFFER_SIZE && status !== "ready") {
            setStatus("buffering")
          } else {
            setStatus("ready")
            setStatusDetail("ASL Translator active")
          }

          if (response.gesture && response.top5) {
            updatePrediction(response.gesture, response.top5)
          } else if (!response.handsDetected) {
            if (predictionSequenceRef.current.length > 0) {
               if (pauseTimeoutRef.current) clearTimeout(pauseTimeoutRef.current)
               processSentence()
            }
          }

          if (isLocal && overlayCanvasRef.current) {
            drawLandmarks(overlayCanvasRef.current, video, response.landmarks || [])
          }

          // Target ~30 FPS for smoother visual tracking
          setTimeout(processFrame, 33)
        } catch (ipcErr: any) {
          if (!active) return
          console.error("[ASL] sendToBackground error:", ipcErr)
          setTimeout(processFrame, 500)
        }
      } catch (e) {
        console.error("Capture error:", e)
        setStatus("error")
        setStatusDetail("Capture failed")
        setTimeout(processFrame, 1000)
      }
    }

    processFrame()
    return () => { active = false }
  }, [video])

  const rect = video.getBoundingClientRect()

  return (
    <div style={{
      position: "fixed",
      top: 0,
      left: 0,
      width: "100vw",
      height: "100vh",
      pointerEvents: "none",
      zIndex: 2147483647
    }}>
      {isLocal && (
        <canvas
          ref={overlayCanvasRef}
          style={{
            position: "absolute",
            pointerEvents: "none",
            zIndex: 2147483646,
          }}
        />
      )}
      {/* Canvas removed as requested */}
      {/* Status Badge Removed as requested */}

      {/* Prediction Result - Movie Captions Style */}
      {(captions.length > 0 || finalSentence) && (
        <div style={{
          position: "absolute",
          bottom: "10%",
          left: "50%",
          transform: "translateX(-50%)",
          background: finalSentence ? "rgba(20, 20, 20, 0.9)" : "rgba(0, 0, 0, 0.6)",
          color: finalSentence ? "#4ade80" : "#f8f9fa",
          padding: "16px 32px",
          borderRadius: "12px",
          fontSize: "32px",
          fontFamily: "system-ui, -apple-system, sans-serif",
          fontWeight: finalSentence ? "800" : "600",
          textShadow: "2px 2px 4px rgba(0, 0, 0, 0.8)",
          zIndex: 2147483647,
          direction: "rtl",
          backdropFilter: "blur(4px)",
          maxWidth: "80%",
          textAlign: "center",
          lineHeight: "1.4",
          transition: "all 0.3s ease",
          border: finalSentence ? "1px solid rgba(74, 222, 128, 0.3)" : "none",
        }}>
          {finalSentence ? finalSentence : captions.join(" ")}
        </div>
      )}
    </div>
  )
}

export default function MeetASLTranslator() {
  const [videoElements, setVideoElements] = useState<HTMLVideoElement[]>([])

  useEffect(() => {
    sendToBackground({ name: "init_offscreen" })
    const cleanup = observeParticipants((video) => {
      setVideoElements(prev => [...prev, video])
    })
    return cleanup
  }, [])

  return (
    <>
      {videoElements.map((video, idx) => (
        <VideoOverlay key={idx} video={video} />
      ))}
    </>
  )
}
