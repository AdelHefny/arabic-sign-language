import type { PlasmoMessaging } from "@plasmohq/messaging"
const OFFSCREEN_DOCUMENT_PATH = "tabs/offscreen.html"

async function setupOffscreenDocument() {
  // @ts-ignore
  if (await chrome.offscreen.hasDocument()) return;
  // @ts-ignore
  await chrome.offscreen.createDocument({
    url: OFFSCREEN_DOCUMENT_PATH,
    // @ts-ignore
    reasons: [chrome.offscreen.Reason.USER_MEDIA],
    justification: "Process video frames using MediaPipe for ASL recognition."
  });
}

const handler: PlasmoMessaging.MessageHandler = async (req, res) => {
  try {
    await setupOffscreenDocument();
    res.send({ success: true });
  } catch (err: any) {
    console.error("[ASL] Failed to create offscreen document:", err.message, err);
    res.send({ success: false, error: err.message });
  }
}

export default handler
