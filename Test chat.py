"""
Quick terminal chat client to test the bot end-to-end before wiring up
Telegram. Make sure the server is running first:

    uvicorn app:app --reload

Then in another terminal:

    python test_chat.py
"""

import sys
import json
import urllib.request

API_URL = "http://127.0.0.1:5000/chat"
USER_ID = "test-user-cli"


def send_message(message: str) -> dict:
    payload = json.dumps({"user_id": USER_ID, "message": message}).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    print("RightBridge test chat. Type 'quit' to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not message:
            continue
        if message.lower() in {"quit", "exit"}:
            break

        try:
            result = send_message(message)
        except Exception as e:
            print(f"[error] Could not reach the server — is uvicorn running? ({e})")
            sys.exit(1)

        print(f"Bot: {result['reply']}")
        if result.get("matched_schemes"):
            print(f"     (matched: {', '.join(result['matched_schemes'])})")
        print()


if __name__ == "__main__":
    main()