// extension/background.js
// Service worker: handles install events and badge updates.

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.set({
    enabled:        true,
    backendUrl:     "ws://localhost:8000/ws/",
    targetFps:      15,
    showFps:        true,
    showConfidence: true,
    fontSize:       "22px",
    opacity:        0.85,
  });
  console.log("[ArSL] Extension installed — defaults set.");
});

// Relay messages between popup and content scripts
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "PING") sendResponse({ type: "PONG" });
});
