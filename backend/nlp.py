"""
backend/nlp.py
==============
Arabic NLP post-processing pipeline.

Responsibilities
----------------
- Sentence buffering & construction from committed sign tokens.
- Rule-based Arabic grammar / spelling correction.
- Duplicate token removal.
- Punctuation insertion.
- Optional AraBERT-powered correction (enabled via config.USE_ARABERT).
"""

from __future__ import annotations

import logging
import re
import time
from typing import List, Optional

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Common Arabic corrections (rule-based)
# ---------------------------------------------------------------------------

# Map common mis-recognised tokens → canonical Arabic words
TOKEN_CORRECTIONS: dict[str, str] = {
    "أ":  "أ",
    "ا":  "ا",
    "ب":  "ب",
    "ت":  "ت",
    "ث":  "ث",
    "ج":  "ج",
    "ح":  "ح",
    "خ":  "خ",
    "د":  "د",
    "ذ":  "ذ",
    # Add domain-specific corrections here
}

# Arabic sentence enders
SENTENCE_ENDERS = {".", "؟", "!", "،"}

# Filler / noise tokens produced when hands are not clear
NOISE_TOKENS = {"", " ", "unknown", "none", "noise"}

# ---------------------------------------------------------------------------
# AraBERT (optional)
# ---------------------------------------------------------------------------

_arabert_pipeline = None

def _load_arabert() -> None:
    global _arabert_pipeline
    if _arabert_pipeline is not None:
        return
    try:
        from transformers import pipeline as hf_pipeline
        _arabert_pipeline = hf_pipeline(
            "fill-mask",
            model=config.ARABERT_MODEL,
            tokenizer=config.ARABERT_MODEL,
        )
        logger.info("AraBERT loaded: %s", config.ARABERT_MODEL)
    except Exception as exc:
        logger.warning("Failed to load AraBERT (%s). Falling back to rule-based.", exc)
        _arabert_pipeline = None


# ---------------------------------------------------------------------------
# Sentence Buffer
# ---------------------------------------------------------------------------

class SentenceBuffer:
    """
    Accumulates committed sign labels and produces corrected Arabic sentences.

    Usage
    -----
    buf = SentenceBuffer()
    buf.push("مرحبا")
    buf.push("كيف")
    buf.push("حالك")
    sentence = buf.flush()   # "مرحبا كيف حالك."
    """

    def __init__(self) -> None:
        self._tokens: List[str]  = []
        self._last_push_ts: float = 0.0

    # ------------------------------------------------------------------
    def push(self, token: str) -> Optional[str]:
        """
        Add a committed sign token to the buffer.

        Returns a completed sentence string if the buffer was auto-flushed
        due to silence timeout or max-word limit; otherwise returns None.
        """
        now = time.monotonic()

        # Auto-flush on long silence
        if (self._tokens
                and (now - self._last_push_ts) > config.SILENCE_TIMEOUT_SECS):
            sentence = self.flush()
            self._tokens = []     # reset after auto-flush
            self._last_push_ts = now
            self.push(token)      # re-push current token into fresh buffer
            return sentence

        token = token.strip()
        if token.lower() in NOISE_TOKENS:
            return None

        # Duplicate suppression: skip if same as previous token
        if self._tokens and self._tokens[-1] == token:
            return None

        self._tokens.append(token)
        self._last_push_ts = now

        # Auto-flush when sentence is long enough
        if len(self._tokens) >= config.MAX_SENTENCE_WORDS:
            return self.flush()

        return None

    # ------------------------------------------------------------------
    def flush(self) -> str:
        """
        Finalise, correct, and return the current sentence then clear buffer.
        """
        if not self._tokens:
            return ""
        raw = " ".join(self._tokens)
        corrected = correct_sentence(raw)
        self._tokens = []
        return corrected

    # ------------------------------------------------------------------
    @property
    def pending_words(self) -> int:
        return len(self._tokens)

    @property
    def preview(self) -> str:
        """Live preview of the buffer (uncorrected)."""
        return " ".join(self._tokens)


# ---------------------------------------------------------------------------
# Correction pipeline
# ---------------------------------------------------------------------------

def correct_sentence(text: str) -> str:
    """
    Full correction pipeline:
      1. Token-level corrections.
      2. Remove consecutive duplicates.
      3. Normalise whitespace.
      4. Add terminal punctuation.
      5. Optional AraBERT correction.
    """
    text = _apply_token_corrections(text)
    text = _remove_consecutive_duplicates(text)
    text = _normalise_whitespace(text)
    text = _add_punctuation(text)

    if config.USE_ARABERT:
        text = _arabert_correct(text)

    return text


def _apply_token_corrections(text: str) -> str:
    tokens = text.split()
    return " ".join(TOKEN_CORRECTIONS.get(t, t) for t in tokens)


def _remove_consecutive_duplicates(text: str) -> str:
    tokens = text.split()
    result: List[str] = []
    prev = None
    for t in tokens:
        if t != prev:
            result.append(t)
        prev = t
    return " ".join(result)


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _add_punctuation(text: str) -> str:
    if text and text[-1] not in SENTENCE_ENDERS:
        text += "."
    return text


def _arabert_correct(text: str) -> str:
    """Optionally run an AraBERT fill-mask pass for grammar refinement."""
    global _arabert_pipeline
    if _arabert_pipeline is None:
        _load_arabert()
    if _arabert_pipeline is None:
        return text
    try:
        # Simple approach: mask one token at a time and keep if higher-confidence
        # A full implementation would use a sliding-window correction strategy.
        # For now we return text unchanged — plug in your correction logic here.
        return text
    except Exception as exc:
        logger.debug("AraBERT correction failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# Convenience: format for subtitle display
# ---------------------------------------------------------------------------

def format_for_subtitle(sentence: str) -> dict:
    """
    Return a dict suitable for JSON serialisation to the Chrome Extension.
    """
    return {
        "text":      sentence,
        "direction": "rtl",
        "lang":      "ar",
        "length":    len(sentence),
    }
