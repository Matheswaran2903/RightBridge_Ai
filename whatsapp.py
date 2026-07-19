"""
whatsapp.py — connects RightBridge to WhatsApp via Meta's free WhatsApp
Cloud API. No cost as long as users message you first: replies inside
the 24-hour "customer service window" that opens after a user messages
you are free, indefinitely, with no message-count limit. You'd only be
billed if you send template/marketing messages that the user did NOT
start the conversation for — which this bot never does.

Setup (all free):
  1. Go to https://developers.facebook.com -> create an app -> add the
     "WhatsApp" product. Meta gives you a free TEST phone number and a
     temporary access token immediately.
  2. Put these in your .env:
       WHATSAPP_TOKEN=<temporary or permanent access token>
       WHATSAPP_PHONE_NUMBER_ID=<from the WhatsApp > API Setup page>
       WHATSAPP_VERIFY_TOKEN=<any string you make up, e.g. rightbridge123>
  3. Your FastAPI server needs a public HTTPS URL for Meta to reach.
       - Local testing: run `ngrok http 8000` (free) and copy the
         https://xxxx.ngrok-free.app URL.
       - Permanent + free hosting: Render.com or Railway.app free tier.
  4. In the Meta App dashboard -> WhatsApp -> Configuration:
       - Webhook URL:  https://<your-public-url>/webhook
       - Verify token: same string as WHATSAPP_VERIFY_TOKEN above
       - Click "Verify and Save", then subscribe to the "messages" field.
  5. In test mode, add up to 5 recipient numbers under "API Setup" (your
     own phone, testers' phones) — message them from those numbers and
     the bot will reply automatically. No credit card needed for this.
  6. To let ANY WhatsApp user message the bot (production), you submit
     the app for review and verify your business — still free, Meta
     doesn't charge for this step either.
"""

import os
import requests
from fastapi import APIRouter, Request, Response
from sqlalchemy.orm import Session

from data import SessionLocal, ConversationLog
from ai import (
    retrieve_relevant_schemes, format_context, get_ai_reply,
    needs_help_centers, needs_emergency_numbers,
    HELP_CENTERS_TEXT, EMERGENCY_NUMBERS_TEXT,
    needs_translation_for_search, translate_for_search,
)

router = APIRouter()

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN", "").strip()
WHATSAPP_PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "").strip()
WHATSAPP_VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN", "rightbridge_verify").strip()
GRAPH_API_VERSION = "v21.0"  # check https://developers.facebook.com/docs/graph-api/changelog for the current version


def _send_whatsapp_message(to: str, text: str) -> None:
    """Sends a plain text reply back to the user via the Cloud API."""
    if not WHATSAPP_TOKEN or not WHATSAPP_PHONE_NUMBER_ID:
        print("WhatsApp not configured: missing WHATSAPP_TOKEN or WHATSAPP_PHONE_NUMBER_ID")
        return

    url = f"https://graph.facebook.com/{GRAPH_API_VERSION}/{WHATSAPP_PHONE_NUMBER_ID}/messages"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}", "Content-Type": "application/json"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text[:4096]},  # WhatsApp's per-message character limit
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=20)
    if resp.status_code >= 300:
        print("WhatsApp send error:", resp.status_code, resp.text)


def _build_history_text(db: Session, user_id: str, max_turns: int = 4) -> str:
    recent = (
        db.query(ConversationLog)
        .filter(ConversationLog.user_id == user_id)
        .order_by(ConversationLog.id.desc())
        .limit(max_turns)
        .all()
    )
    if not recent:
        return ""
    recent.reverse()
    lines = []
    for turn in recent:
        lines.append(f"User: {turn.user_message}")
        lines.append(f"RightBridge: {turn.bot_reply}")
    return "\n".join(lines)


@router.get("/webhook")
def verify_webhook(request: Request):
    """Meta calls this once when you click 'Verify and Save' in the dashboard."""
    params = request.query_params
    if params.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN:
        return Response(content=params.get("hub.challenge", ""), media_type="text/plain")
    return Response(content="Verification failed", status_code=403)


@router.post("/webhook")
async def receive_message(request: Request):
    """Meta POSTs every incoming WhatsApp message here."""
    body = await request.json()
    try:
        entry = body["entry"][0]
        change = entry["changes"][0]["value"]
        messages = change.get("messages")
        if not messages:
            return {"status": "ignored"}  # e.g. delivery/read receipts, not a user message

        msg = messages[0]
        from_number = msg["from"]  # user's WhatsApp number, used as the user_id
        user_text = msg.get("text", {}).get("body", "").strip()
        if not user_text:
            return {"status": "ignored"}  # non-text message (image, voice note, etc.)

        db: Session = SessionLocal()
        try:
            search_query = user_text
            if needs_translation_for_search(user_text):
                search_query = translate_for_search(user_text)
            schemes = retrieve_relevant_schemes(db, search_query)
            context = format_context(schemes)
            history_text = _build_history_text(db, from_number)
            reply = get_ai_reply(user_text, context, history_text)

            if needs_emergency_numbers(user_text):
                reply += EMERGENCY_NUMBERS_TEXT
            elif needs_help_centers(user_text):
                reply += HELP_CENTERS_TEXT

            db.add(ConversationLog(
                user_id=from_number,
                user_message=user_text,
                bot_reply=reply,
                matched_scheme_names=", ".join(s.name for s in schemes),
            ))
            db.commit()
        finally:
            db.close()

        _send_whatsapp_message(from_number, reply)
        return {"status": "ok"}

    except Exception as e:
        print("Webhook error:", e)
        return {"status": "error"}