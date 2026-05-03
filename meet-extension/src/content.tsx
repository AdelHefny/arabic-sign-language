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
  const [prediction, setPrediction] = useState("")
  const [status, setStatus] = useState<AppStatus>("loading")
  const [statusDetail, setStatusDetail] = useState("Initializing...")
  const [bufferProgress, setBufferProgress] = useState(0)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const BUFFER_SIZE = 30
  const participantId = useRef(Math.random().toString(36).slice(2)).current

  const updatePrediction = (text: string) => {
    setPrediction(text)
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(() => setPrediction(""), 3000)
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

      captureCanvas.width = 320
      captureCanvas.height = Math.round(320 * (video.videoHeight / video.videoWidth))

      try {
        captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height)
        const imageData = captureCtx.getImageData(0, 0, captureCanvas.width, captureCanvas.height)

        sendToBackground({
          name: "process_frame",
          body: {
            participantId,
            imageData: {
              data: Array.from(imageData.data),
              width: imageData.width,
              height: imageData.height
            }
          }
        }).then((response: any) => {
          if (!active || !response) return

          // Canvas clearing removed as visible canvas is gone

          // --- Update UI state based on response ---
          if (response.status === "loading") {
            setStatus("loading")
            setStatusDetail("Loading model...")
            return
          }

          if (response.error) {
            setStatus("error")
            setStatusDetail(response.error)
            return
          }

          if (response.landmarks) {
            // Landmarks drawing removed as requested
          }

          const progress = response.bufferProgress ?? 0
          setBufferProgress(progress)

          if (!response.handsDetected) {
            setStatus("no_hands")
          } else if (progress < BUFFER_SIZE) {
            setStatus("buffering")
          } else {
            setStatus("ready")
            setStatusDetail("ASL Translator active")
          }

          if (response.gesture) {
            console.log("[ASL] Prediction:", response.gesture)
            updatePrediction(response.gesture)
          } else if (response.handsDetected) {
            // Log every ~2 seconds to avoid spamming if no prediction
            if (Math.random() < 0.05) console.log("[ASL] Tracking hands, awaiting prediction window...")
          }
        })
      } catch (e) {
        console.error("Capture error:", e)
        setStatus("error")
        setStatusDetail("Capture failed")
      }

      setTimeout(processFrame, 66)
    }

    processFrame()
    return () => { active = false }
  }, [video])

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
      {/* Canvas removed as requested */}

      {/* Status Badge */}
      <StatusBadge
        status={status}
        detail={statusDetail}
        bufferProgress={bufferProgress}
        bufferSize={BUFFER_SIZE}
      />

      {/* Prediction Result */}
      {status === "ready" && (
        <div style={{
          position: "absolute",
          top: "40px", // Higher and sleeker
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(10, 10, 15, 0.95)",
          color: "#00f2ff",
          padding: "8px 20px",
          borderRadius: "12px",
          fontSize: "22px", // Smaller, professional size
          fontFamily: "system-ui, -apple-system, sans-serif",
          fontWeight: "600",
          border: "1px solid rgba(0, 242, 255, 0.5)",
          boxShadow: "0 4px 15px rgba(0, 0, 0, 0.5)",
          display: "flex",
          alignItems: "center",
          gap: "10px",
          zIndex: 2147483647,
          direction: "rtl",
          backdropFilter: "blur(8px)",
        }}>
          <span style={{ fontSize: "16px" }}>🤟</span>
          {prediction || "..."}
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
