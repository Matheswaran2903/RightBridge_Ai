# RightBridge — Chatbot Backend (Step 1)

Down to **4 Python files** now — easier to drag into VS Code.

| File | What's inside |
|---|---|
| `app.py` | FastAPI app: request/response schemas + all endpoints (`/chat`, `/schemes`, `/health`) |
| `data.py` | Database setup, `Scheme`/`ConversationLog` tables, 10 seed schemes, Hugging Face dataset loader |
| `ai.py` | Keyword retrieval + Gemini API call (system prompt with grounding rules) |
| `test_chat.py` | Terminal chat client — talk to your bot without Postman/Telegram |

Plus: `requirements.txt`, `.env.example`, `.gitignore`, this `README.md`.

## Setup (one time)

1. Create a new empty folder, e.g. `rightbridge`, and put all the downloaded files inside it directly (not in a subfolder).
2. Open that folder in VS Code (File → Open Folder).
3. Open the integrated terminal (View → Terminal) and run:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt --break-system-packages
```

4. Get a **free** Gemini API key: https://aistudio.google.com/apikey
5. Copy `.env.example` → `.env` (same folder) and paste your key:
```
GEMINI_API_KEY=AIza...
```

## Run it

**Terminal 1 — start the server:**
```bash
uvicorn app:app --reload
```
On first startup it auto-loads the Hugging Face dataset `shrijayan/gov_myscheme`. If that fails for any reason (no internet, dataset renamed, rate limited) it silently falls back to the 10 built-in seed schemes — the app never breaks.

Check it worked:
```
http://127.0.0.1:5000/health
```
→ `{"status":"ok","schemes_loaded":723}` (or `10` if on seed fallback)

**Terminal 2 (click `+` in the VS Code terminal panel) — chat with it:**
```bash
python test_chat.py
```
Or open **http://127.0.0.1:5000/docs** in your browser and try `/chat` directly — no extra tools needed.

## Reload fresh data from Hugging Face

```bash
python data.py --reset
```

## If Hugging Face field names don't match

`data.py` prints the real column names on every load attempt:
```
[data] Sample raw row keys: ['scheme_name', 'eligibility_description', ...]
```
If scheme details look empty, paste me that line and I'll update the `FIELD_ALIASES` dict in `data.py` to match.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `"The AI service isn't configured yet..."` in replies | `.env` missing or `GEMINI_API_KEY` empty/wrong |
| `schemes_loaded: 10` in `/health` | Running on seed fallback — fine for testing, run `python data.py --reset` once you have internet |
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt --break-system-packages` with the venv activated |

## Next step

Once this works well locally: deploy to Hugging Face Spaces or Render (free, no card), then wire up Telegram. Ask for that next when ready.