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


def needs_translation_for_search(text: str) -> bool:
    """True if the query has no usable English keywords (e.g. pure Tamil/
    Hindi/other script) — the DB is in English, so keyword search would
    otherwise return random unrelated schemes instead of relevant ones."""
    return not _tokenize(text)


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
# GNANI.AI VOICE — TTS (text -> speech) and STT (speech -> text) for Indian
# languages, using the official `gnani-vachana` SDK (pip install gnani-vachana).
# NOTE: the pip package name is "gnani-vachana" but the Python import is
# just `gnani` (confirmed by inspecting the actual package contents).
# Requires GNANI_API_KEY in .env (get one at https://gnani.ai).
# ---------------------------------------------------------------------------

GNANI_API_KEY = os.environ.get("GNANI_API_KEY", "").strip()

_tts_client = None
_stt_client = None
_gnani_error = None

if GNANI_API_KEY:
    try:
        from gnani.tts import GnaniTTSClient
        from gnani.stt import GnaniSTTClient
        _tts_client = GnaniTTSClient(api_key=GNANI_API_KEY)
        _stt_client = GnaniSTTClient(api_key=GNANI_API_KEY)
    except Exception as e:  # pragma: no cover
        _gnani_error = str(e)
else:
    _gnani_error = "GNANI_API_KEY is missing. Add it to your .env file."


def gnani_text_to_speech(text: str, voice: str = "Pranav") -> bytes:
    """Synthesises speech with Gnani TTS (timbre-v2.0) and returns WAV audio
    bytes. Valid timbre-v2.0 voices: Pranav, Kaveri, Shubhra, Deepak.
    Raises RuntimeError on failure — the caller (app.py endpoint) turns
    that into a clean HTTP error."""
    if _tts_client is None:
        raise RuntimeError(f"Gnani TTS isn't configured: {_gnani_error}")
    try:
        return _tts_client.synthesize(text, voice=voice)
    except Exception as e:
        raise RuntimeError(f"Gnani TTS error: {e}")


def gnani_speech_to_text(audio_bytes: bytes, language_code: str = "hi-IN", filename: str = "audio.wav") -> str:
    """Transcribes audio bytes with Gnani STT. language_code must be one of:
    en-IN, hi-IN, gu-IN, ta-IN, kn-IN, te-IN, mr-IN, bn-IN, ml-IN, pa-IN.
    Raises RuntimeError on failure."""
    if _stt_client is None:
        raise RuntimeError(f"Gnani STT isn't configured: {_gnani_error}")
    try:
        result = _stt_client.transcribe_bytes(audio_bytes, filename=filename, language_code=language_code)
        return result.get("transcript", "")
    except Exception as e:
        raise RuntimeError(f"Gnani STT error: {e}")


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


TRANSLATE_SYSTEM_INSTRUCTION = """You translate short questions into English \
so they can be used to search an English-language database. Reply with ONLY \
the English translation — no extra words, no explanation, no quotes."""


def translate_for_search(text: str) -> str:
    """Translates a non-English query into English, used ONLY to pick which
    schemes to retrieve from the database. The user-facing reply is still
    generated in the user's original language separately. Falls back to the
    original text if translation fails."""
    if _client is None:
        return text
    try:
        translated = _call_gemini(
            f"Translate to English:\n\n{text}",
            TRANSLATE_SYSTEM_INSTRUCTION,
            max_output_tokens=100,
        )
        # If the AI service errored out, _call_gemini returns a long
        # human-readable error string — don't use that as a search query.
        if translated and "Sorry" not in translated and "debug:" not in translated:
            return translated
    except Exception:
        pass
    return text


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


def get_simple_explanation(original_text: str) -> str:
    """FEATURE 3: 'Explain Simply' — rewrites an existing bot reply in
    simpler language, same facts, same language/script."""
    if not original_text.strip():
        return "There is nothing to simplify yet."

    prompt = f"Rewrite this in simple language:\n\n{original_text}"
    return _call_gemini(prompt, SIMPLIFY_SYSTEM_INSTRUCTION, max_output_tokens=600)