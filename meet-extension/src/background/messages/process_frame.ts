import type { PlasmoMessaging } from "@plasmohq/messaging"

const handler: PlasmoMessaging.MessageHandler = async (req, res) => {
  // Forward the frame to the offscreen document using native chrome messaging
  // because offscreen documents are not technically content scripts and we
  // need to explicitly target them.
  await new Promise<void>((resolve) => {
    chrome.runtime.sendMessage(
      { target: "offscreen", type: "PROCESS_FRAME", participantId: req.body.participantId, dataUrl: req.body.dataUrl, width: req.body.width, height: req.body.height },
      (response) => {
        if (chrome.runtime.lastError) {
          console.error("[ASL] Forwarding error:", chrome.runtime.lastError.message);
          res.send({ error: chrome.runtime.lastError.message });
        } else {
          res.send(response);
        }
        resolve();
      }
    );
  });
}

export default handler
