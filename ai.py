"""
ai.py — all the "intelligence" of RightBridge in one place:
  1. retrieve_relevant_schemes() / format_context()  -> keyword search over the DB
  2. get_ai_reply()                                   -> calls Gemini with grounded context
  3. get_simple_explanation()                         -> "Explain Simply" feature
  4. needs_help_centers() / needs_emergency_numbers()  -> keyword triggers for
     Nearby Help Centers + Emergency Numbers features
"""

import os
import re
import time
from typing import List
from dotenv import load_dotenv
from sqlalchemy.orm import Session

from data import Scheme  # Scheme model lives in data.py

load_dotenv()

# ---------------------------------------------------------------------------
# 1. RETRIEVAL — simple keyword-overlap search (fast, free, no embeddings needed)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the", "a", "an", "is", "am", "are", "was", "were", "what", "which",
    "who", "whom", "how", "do", "does", "did", "can", "could", "should",
    "would", "will", "for", "of", "to", "in", "on", "at", "and", "or",
    "my", "me", "i", "you", "your", "please", "tell", "about", "get",
    "eligible", "eligibility", "scheme", "schemes",
}


def _tokenize(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return [w for w in words if len(w) > 2 and w not in _STOPWORDS]


def retrieve_relevant_schemes(db: Session, query: str, limit: int = 5) -> List[Scheme]:
    """Returns up to `limit` schemes most relevant to the query."""
    keywords = _tokenize(query)
    all_schemes = db.query(Scheme).all()

    if not all_schemes:
        return []
    if not keywords:
        return all_schemes[:limit]

    scored = []
    for s in all_schemes:
        blob = " ".join(filter(None, [s.name, s.category, s.eligibility, s.benefits])).lower()
        score = sum(1 for kw in keywords if kw in blob)
        if score > 0:
            scored.append((score, s))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    top = [s for _, s in scored[:limit]]
    return top if top else all_schemes[:limit]


_CORRUPT_RUN_RE = re.compile(r"(?:&.){3,}&?")


def clean_scheme_text(text):
    """Some rows in the source dataset have corrupted text where a stray
    '&' is interleaved between every real character (e.g. '&w&i&l&l&').
    This finds each corrupted RUN (3+ repeats) and strips the '&' out of
    just that run, leaving the rest of the string untouched. Mirrors the
    same fix already used on the frontend for PDF downloads, but applied
    here so the AI never sees/repeats the raw corrupted text either."""
    if not text:
        return text
    s = str(text)
    s = _CORRUPT_RUN_RE.sub(lambda m: m.group(0).replace("&", ""), s)
    s = s.replace("þÿ", "")  # leftover UTF-16 BOM artifacts
    return s


def format_context(schemes: List[Scheme]) -> str:
    if not schemes:
        return ""
    lines = []
    for s in schemes:
        lines.append(
            f"- {clean_scheme_text(s.name)} (category: {clean_scheme_text(s.category) or 'N/A'})\n"
            f"  Eligibility: {clean_scheme_text(s.eligibility) or 'Not specified'}\n"
            f"  Benefits: {clean_scheme_text(s.benefits) or 'Not specified'}\n"
            f"  How to apply: {clean_scheme_text(s.how_to_apply) or 'Not specified'}\n"
            f"  Documents required: {clean_scheme_text(s.documents_required) or 'Not specified'}\n"
            f"  Official website: {s.official_link or 'Not available'}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# FEATURE 1: NEARBY HELP CENTERS
# FEATURE 4: EMERGENCY NUMBERS
# Static, keyword-triggered blocks. No location API needed — kept generic
# so it works anywhere in India; edit the numbers below if you have
# local/state-specific ones.
# ---------------------------------------------------------------------------

HELP_CENTERS_TEXT = (
    "\n\n📍 Nearby Support\n"
    "📍 Nearest Common Service Centre (CSC)\n"
    "📍 Labour Welfare Office\n"
    "📞 eShram Helpline: 14434"
)

EMERGENCY_NUMBERS_TEXT = (
    "\n\n🚨 Emergency Numbers\n"
    "📞 Labour Helpline: 1800-11-1000\n"
    "📞 National Helpline: 1800-11-1000\n"
    "🏢 State Labour Office: contact your district Labour Commissioner's office\n"
    "📞 Emergency (Police/Ambulance): 112"
)

_APPLY_KEYWORDS = {
    "apply", "application", "register", "registration", "enroll", "enrol",
    "eshram", "csc", "center", "centre", "office", "visit",
}

_EMERGENCY_KEYWORDS = {
    "salary", "wage", "wages", "unpaid", "employer", "harassment", "abuse",
    "accident", "injury", "emergency", "fired", "terminated", "exploit",
    "cheated", "fraud", "threat", "threatened", "assault",
}


def needs_help_centers(message: str) -> bool:
    """True if the user's question is about applying/registering — trigger
    for suggesting nearby CSC / Labour Welfare Office."""
    tokens = set(_tokenize(message))
    return bool(tokens & _APPLY_KEYWORDS)


def needs_emergency_numbers(message: str) -> bool:
    """True if the message suggests an urgent labour-rights problem (unpaid
    salary, harassment, injury, etc.) — trigger for emergency contacts."""
    lower = message.lower()
    return any(kw in lower for kw in _EMERGENCY_KEYWORDS)


# ---------------------------------------------------------------------------
# 2. LLM CALL — Gemini, grounded on retrieved context only
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
MODEL_NAME = "gemini-3.5-flash"  # current supported model as of July 2026

_client = None
_client_error = None

if GEMINI_API_KEY:
    try:
        from google import genai
        _client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:  # pragma: no cover
        _client_error = str(e)
else:
    _client_error = "GEMINI_API_KEY is missing. Add it to your .env file."


SYSTEM_INSTRUCTION = """You are RightBridge, a friendly assistant that helps Indian \
internal migrant workers (construction workers, factory laborers, domestic help, \
gig workers) understand government welfare schemes and their basic labor rights.

Rules you must always follow:
1. Reply in the SAME language and script the user wrote in. If they write in \
   Tamil, reply in Tamil. If they write in Hinglish, reply in Hinglish. If unsure, \
   default to simple English.
2. Base your answer ONLY on the "Scheme context" provided below. Never invent \
   scheme names, amounts, deadlines, or eligibility rules that are not in the context.
3. If the context does not contain a clear answer, say so honestly and suggest the \
   user visit the nearest Common Service Centre (CSC) or call the eShram helpline \
   (14434) instead of guessing.
4. Keep answers short, simple, and in plain everyday language — avoid legal or \
   bureaucratic jargon. Use short sentences and simple words, since many users have \
   low literacy.
5. Whenever you name a specific scheme in your answer, ALWAYS include both: (a) how \
   to apply, in 1-2 concrete steps, and (b) its official website link — both are given \
   in the Scheme context below. Never omit the link if one is listed as available; \
   never invent a link if it says "Not available".
6. If "Conversation so far" is provided below, use it to remember facts the user \
   already told you (e.g. their age, occupation, state, or that they are a student) \
   so you don't ask them to repeat themselves.
7. If the user asks how to apply, what documents are needed, or similar, list the \
   "Documents required" from the Scheme context as a checklist using a ✓ before each \
   item (e.g. "✓ Aadhaar Card"). Never invent a document that isn't in the context.
"""

SIMPLIFY_SYSTEM_INSTRUCTION = """You rewrite text for Indian migrant workers with \
low literacy. Rewrite the given text in the SAME language/script it is already in, \
using very short sentences, everyday simple words, and no jargon. Keep all facts, \
numbers, links, and instructions exactly the same — do not add or remove information, \
only simplify the language. Keep any emoji/bullet formatting from the original."""


def _call_gemini(prompt: str, system_instruction: str, max_output_tokens: int = 800) -> str:
    """Shared Gemini call with retry-on-transient-error logic. Never raises."""
    if _client is None:
        return (
            "The AI service isn't configured yet (missing or invalid Gemini API key). "
            "Please check the .env file. "
            f"[debug: {_client_error}]"
        )

    delay = 2
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            from google.genai import types

            response = _client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.3,
                    max_output_tokens=max_output_tokens,
                ),
            )
            text = getattr(response, "text", None)
            if not text:
                return "Sorry, I couldn't generate a reply just now. Please try again."
            return text.strip()

        except Exception as e:
            err = str(e)
            is_transient = "503" in err or "UNAVAILABLE" in err or "429" in err
            if is_transient and attempt < max_retries:
                time.sleep(delay)
                delay *= 2
                continue
            return (
                "Sorry, there was a problem reaching the AI service. "
                "Please try again in a moment. "
                f"[debug: {e}]"
            )


def get_ai_reply(user_message: str, context_block: str, history_text: str = "") -> str:
    """Calls Gemini with the user's message, retrieved context, and optional
    recent conversation history (for multi-turn memory). Never raises."""
    if not context_block.strip():
        context_block = "No matching schemes were found in the database for this query."

    history_block = f"Conversation so far:\n{history_text}\n\n" if history_text.strip() else ""

    prompt = (
        f"{history_block}"
        f"Scheme context (only use facts from here):\n{context_block}\n\n"
        f"User question: {user_message}"
    )

    return _call_gemini(prompt, SYSTEM_INSTRUCTION)


# ---------------------------------------------------------------------------
# GNANI.AI VOICE — TTS (text -> speech) and STT (speech -> text) for Indian
# languages. Requires GNANI_API_KEY in .env (get one at https://gnani.ai).
# For STT, GNANI_ORGANIZATION_ID and GNANI_USER_ID may also be required —
# check your Gnani dashboard/docs, since these vary by plan.
# ---------------------------------------------------------------------------

import base64
import requests

GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "").strip()
GNANI_TTS_URL = "https://api.vachana.ai/api/v1/tts/sse"


def gnani_text_to_speech(text: str, language: str = "hi", voice: str = "Karan") -> bytes:
    """Calls Gnani's TTS REST API and returns raw audio bytes (mp3).
    Raises RuntimeError on failure — the caller (app.py endpoint) turns
    that into a clean HTTP error."""
    if not GNANI_API_KEY:
        raise RuntimeError("GNANI_API_KEY is missing. Add it to your .env file.")

    payload = {
        "audio_config": {
            "bitrate": "192k",
            "container": "mp3",
            "encoding": "linear_pcm",
            "num_channels": 1,
            "sample_rate": 44100,
            "sample_width": 2,
        },
        "model": "vachana-voice-v3",
        "text": text,
        "voice": voice,
        "language": language,
    }
    headers = {"X-API-Key-ID": GNANI_API_KEY, "Content-Type": "application/json"}

    resp = requests.post(GNANI_TTS_URL, json=payload, headers=headers, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Gnani TTS error {resp.status_code}: {resp.text}")

    # The /sse endpoint streams chunks; for a simple non-streaming call the
    # response body is the audio bytes directly. If your account only
    # supports the streaming form, switch to the gnani-vachana SDK instead
    # (pip install gnani-vachana) which handles SSE parsing for you.
    return resp.content


def gnani_speech_to_text(audio_bytes: bytes, language: str = "hi-IN") -> str:
    """Transcribes audio using Gnani's Vachana STT via their official SDK.
    Requires: pip install gnani-vachana
    Raises RuntimeError on failure."""
    if not GNANI_API_KEY:
        raise RuntimeError("GNANI_API_KEY is missing. Add it to your .env file.")

    try:
        from gnani_vachana import SttClient  # official Gnani SDK
    except ImportError:
        raise RuntimeError(
            "The gnani-vachana package isn't installed. Run: "
            "pip install gnani-vachana --break-system-packages"
        )

    client = SttClient(
        api_key=GNANI_API_KEY,
        organization_id=os.environ.get("GNANI_ORGANIZATION_ID", "").strip() or None,
        user_id=os.environ.get("GNANI_USER_ID", "").strip() or None,
    )
    result = client.recognize(audio=audio_bytes, language=language)
    # NOTE: verify the exact response shape against the SDK's current docs —
    # this assumes a `.text` attribute on the result, which is the common
    # pattern for their REST-based non-streaming recognize() call.
    return getattr(result, "text", "") or ""


def get_simple_explanation(original_text: str) -> str:
    """FEATURE 3: 'Explain Simply' — rewrites an existing bot reply in
    simpler language, same facts, same language/script."""
    if not original_text.strip():
        return "There is nothing to simplify yet."

    prompt = f"Rewrite this in simple language:\n\n{original_text}"
    return _call_gemini(prompt, SIMPLIFY_SYSTEM_INSTRUCTION, max_output_tokens=600)