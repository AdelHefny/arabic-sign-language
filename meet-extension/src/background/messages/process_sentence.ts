import type { PlasmoMessaging } from "@plasmohq/messaging"

const MODELS = [
  "google/gemma-4-26b-a4b-it:free",
  "nvidia/nemotron-3-ultra-550b-a55b:free",
  "google/gemma-4-31b-it:free",
  "qwen/qwen3-next-80b-a3b-instruct:free",
  "openai/gpt-oss-120b:free"
]

const handler: PlasmoMessaging.MessageHandler = async (req, res) => {
  const { sequence } = req.body

  if (!sequence || sequence.length === 0) {
    res.send({ sentence: "" })
    return
  }

  // Format the sequence for the prompt
  const sequenceText = sequence.map((top5: any[], idx: number) => {
    const words = top5.map((p, i) => `${i + 1}. ${p.word}`).join(", ")
    return `Step ${idx + 1}: ${words}`
  }).join("\n")

  const prompt = `You are an expert Arabic Sign Language translator.
You will receive a time-series sequence of top-5 predicted Arabic words from a vision model.
The sequence might contain repeated words, dropped frames, or slightly incorrect predictions.
Your task is to analyze these predictions and output a SINGLE, grammatically correct, natural conversational Arabic sentence that represents the user's intended meaning.
Do NOT output any explanations, translations to other languages, or extra text. ONLY output the final Arabic sentence.

Sequence:
${sequenceText}

Final Arabic sentence:`

  // Plasmo exposes env vars prefixed with PLASMO_PUBLIC_. 
  // We'll also try OPENROUTER_API in case it's configured to inject it.
  const apiKey = process.env.PLASMO_PUBLIC_OPENROUTER_API || process.env.OPENROUTER_API

  if (!apiKey) {
    res.send({ error: "OpenRouter API key not found. Please set PLASMO_PUBLIC_OPENROUTER_API in .env" })
    return
  }

  let success = false
  let errorMsg = "All models failed."

  for (const model of MODELS) {
    try {
      const response = await fetch("https://openrouter.ai/api/v1/chat/completions", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${apiKey}`,
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          model: model,
          messages: [
            { role: "user", content: prompt }
          ],
          temperature: 0.1,
          max_tokens: 100
        })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      if (data && data.choices && data.choices.length > 0) {
        const sentence = data.choices[0].message.content.trim()
        res.send({ sentence, modelUsed: model })
        success = true
        break
      }
    } catch (err: any) {
      console.error(`[ASL] Model ${model} failed:`, err)
      errorMsg = err.message
      continue // Try next model
    }
  }

  if (!success) {
    res.send({ error: errorMsg })
  }
}

export default handler
