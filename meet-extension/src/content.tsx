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
  const [captions, setCaptions] = useState<string[]>([])
  const [status, setStatus] = useState<AppStatus>("loading")
  const [statusDetail, setStatusDetail] = useState("Initializing...")
  const [bufferProgress, setBufferProgress] = useState(0)
  const timeoutRef = useRef<NodeJS.Timeout | null>(null)
  const wordLimitRef = useRef(Math.floor(Math.random() * 5) + 6) // Random between 6 and 10
  const BUFFER_SIZE = 30
  const participantId = useRef(Math.random().toString(36).slice(2)).current

  const updatePrediction = (text: string) => {
    setCaptions(prev => {
      const newCaptions = [...prev, text]
      if (newCaptions.length > wordLimitRef.current) {
        return newCaptions.slice(newCaptions.length - wordLimitRef.current)
      }
      return newCaptions
    })
    if (timeoutRef.current) clearTimeout(timeoutRef.current)
    timeoutRef.current = setTimeout(() => {
      setCaptions([])
      wordLimitRef.current = Math.floor(Math.random() * 5) + 6
    }, 5000)
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

          if (response.gesture) {
            updatePrediction(response.gesture)
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
      {/* Canvas removed as requested */}
      {/* Status Badge Removed as requested */}

      {/* Prediction Result - Movie Captions Style */}
      {captions.length > 0 && (
        <div style={{
          position: "absolute",
          bottom: "10%",
          left: "50%",
          transform: "translateX(-50%)",
          background: "rgba(0, 0, 0, 0.75)",
          color: "#f8f9fa",
          padding: "16px 32px",
          borderRadius: "12px",
          fontSize: "32px",
          fontFamily: "system-ui, -apple-system, sans-serif",
          fontWeight: "700",
          textShadow: "2px 2px 4px rgba(0, 0, 0, 0.8)",
          zIndex: 2147483647,
          direction: "rtl",
          backdropFilter: "blur(4px)",
          maxWidth: "80%",
          textAlign: "center",
          lineHeight: "1.4",
          transition: "all 0.3s ease",
        }}>
          {captions.join(" ")}
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
