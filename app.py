"""
app.py — the FastAPI application itself. Run with:

    uvicorn app:app --reload

Then open http://127.0.0.1:8000 for the chat UI, or /docs to test endpoints directly.
"""

import os
from typing import List, Optional
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from data import (
    SessionLocal, init_db, Scheme, ConversationLog, Feedback, User,
    hash_password, verify_password, populate_database,
)
from ai import (
    retrieve_relevant_schemes, format_context, get_ai_reply,
    get_simple_explanation, needs_help_centers, needs_emergency_numbers,
    HELP_CENTERS_TEXT, EMERGENCY_NUMBERS_TEXT, clean_scheme_text,
    gnani_text_to_speech, gnani_speech_to_text,
    needs_translation_for_search, translate_for_search,
)
from whatsapp import router as whatsapp_router

# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="Unique id per user (e.g. Telegram chat id)")
    message: str = Field(..., min_length=1, description="The user's message, any language")


class SchemeMatch(BaseModel):
    id: int
    name: str
    official_link: Optional[str] = None


class ChatResponse(BaseModel):
    reply: str
    matched_schemes: List[SchemeMatch] = []


class SchemeOut(BaseModel):
    id: int
    name: str
    category: Optional[str] = None
    eligibility: Optional[str] = None
    benefits: Optional[str] = None
    how_to_apply: Optional[str] = None
    official_link: Optional[str] = None
    documents_required: Optional[str] = None

    class Config:
        from_attributes = True


class FeedbackRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    scheme_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5, description="1 to 5 stars")
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str


class SignupRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6, description="At least 6 characters")


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class AuthResponse(BaseModel):
    status: str
    username: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1)
    voice: str = Field(default="Pranav", description="Pranav, Kaveri, Shubhra, or Deepak")


class STTResponse(BaseModel):
    text: str


class SimplifyRequest(BaseModel):
    text: str = Field(..., min_length=1, description="The bot reply to rewrite simply")


class SimplifyResponse(BaseModel):
    reply: str


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RightBridge API",
    description="Conversational AI for Indian migrant worker welfare schemes.",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(whatsapp_router)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def startup():
    init_db()
    populate_database(reset=False)  # auto-loads data on first run


def _build_history_text(db: Session, user_id: str, max_turns: int = 4) -> str:
    """Pulls the last few turns for this user so Gemini can remember facts
    they already stated (age, occupation, state, student status, etc)."""
    recent = (
        db.query(ConversationLog)
        .filter(ConversationLog.user_id == user_id)
        .order_by(ConversationLog.id.desc())
        .limit(max_turns)
        .all()
    )
    if not recent:
        return ""
    recent.reverse()  # oldest first, so it reads like a real conversation
    lines = []
    for turn in recent:
        lines.append(f"User: {turn.user_message}")
        lines.append(f"RightBridge: {turn.bot_reply}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"status": "RightBridge API is running", "docs": "/docs"}


@app.get("/login.html")
def login_page():
    if os.path.exists("login.html"):
        return FileResponse("login.html")
    raise HTTPException(status_code=404, detail="login.html not found in project folder.")


@app.get("/signup.html")
def signup_page():
    if os.path.exists("signup.html"):
        return FileResponse("signup.html")
    raise HTTPException(status_code=404, detail="signup.html not found in project folder.")


@app.post("/signup", response_model=AuthResponse)
def signup(request: SignupRequest, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == request.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken.")

    pwd_hash, salt = hash_password(request.password)
    user = User(username=request.username, password_hash=pwd_hash, password_salt=salt)
    db.add(user)
    db.commit()
    return AuthResponse(status="ok", username=request.username)


@app.post("/login", response_model=AuthResponse)
def login(request: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == request.username).first()
    if not user or not verify_password(request.password, user.password_hash, user.password_salt):
        raise HTTPException(status_code=401, detail="Incorrect username or password.")
    return AuthResponse(status="ok", username=user.username)


@app.get("/health")
def health(db: Session = Depends(get_db)):
    return {"status": "ok", "schemes_loaded": db.query(Scheme).count()}


@app.get("/schemes", response_model=List[SchemeOut])
def list_schemes(db: Session = Depends(get_db), limit: int = 50):
    return db.query(Scheme).limit(limit).all()


@app.get("/scheme/{scheme_id}", response_model=SchemeOut)
def get_scheme(scheme_id: int, db: Session = Depends(get_db)):
    """Full details for one scheme — used by the frontend for the PDF download
    and favorites detail view."""
    scheme = db.query(Scheme).filter(Scheme.id == scheme_id).first()
    if not scheme:
        raise HTTPException(status_code=404, detail="Scheme not found.")

    # Build the response manually (rather than mutating `scheme` directly)
    # so cleaned text is never accidentally written back to the database.
    return SchemeOut(
        id=scheme.id,
        name=clean_scheme_text(scheme.name),
        category=clean_scheme_text(scheme.category),
        eligibility=clean_scheme_text(scheme.eligibility),
        benefits=clean_scheme_text(scheme.benefits),
        how_to_apply=clean_scheme_text(scheme.how_to_apply),
        official_link=scheme.official_link,
        documents_required=clean_scheme_text(scheme.documents_required),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, db: Session = Depends(get_db)):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    search_query = request.message
    if needs_translation_for_search(request.message):
        search_query = translate_for_search(request.message)
    schemes = retrieve_relevant_schemes(db, search_query)
    context = format_context(schemes)
    history_text = _build_history_text(db, request.user_id)
    reply = get_ai_reply(request.message, context, history_text)

    # FEATURE 4: Emergency Numbers takes priority (e.g. unpaid salary, abuse)
    # FEATURE 1: Nearby Help Centers (e.g. how/where to apply)
    if needs_emergency_numbers(request.message):
        reply += EMERGENCY_NUMBERS_TEXT
    elif needs_help_centers(request.message):
        reply += HELP_CENTERS_TEXT

    matched = [
        SchemeMatch(id=s.id, name=s.name, official_link=s.official_link)
        for s in schemes
    ]

    db.add(ConversationLog(
        user_id=request.user_id,
        user_message=request.message,
        bot_reply=reply,
        matched_scheme_names=", ".join(s.name for s in schemes),
    ))
    db.commit()

    return ChatResponse(reply=reply, matched_schemes=matched)


@app.post("/simplify", response_model=SimplifyResponse)
def simplify(request: SimplifyRequest):
    """FEATURE 3: 'Explain Simply' button — rewrites a given bot reply in
    simpler language, same facts, same language."""
    simple_reply = get_simple_explanation(request.text)
    return SimplifyResponse(reply=simple_reply)


@app.post("/tts")
def text_to_speech(request: TTSRequest):
    """Gnani.ai TTS — returns an mp3 audio file for the given text."""
    try:
        audio_bytes = gnani_text_to_speech(request.text, request.voice)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return Response(content=audio_bytes, media_type="audio/wav")


@app.post("/stt", response_model=STTResponse)
async def speech_to_text(audio: UploadFile = File(...), language: str = "hi-IN"):
    """Gnani.ai STT — accepts an uploaded audio file, returns transcribed text."""
    audio_bytes = await audio.read()
    try:
        text = gnani_speech_to_text(audio_bytes, language)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return STTResponse(text=text)


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest, db: Session = Depends(get_db)):
    db.add(Feedback(
        user_id=request.user_id,
        scheme_id=request.scheme_id,
        rating=request.rating,
        comment=request.comment,
    ))
    db.commit()
    return FeedbackResponse(status="ok")