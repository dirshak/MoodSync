"""Text -> (valence, arousal, one-line reply) via an LLM.

Tries Groq first (fast hosted API, needs GROQ_API_KEY). Falls back to a local
Ollama server if Groq is unavailable or fails. Falls back to a neutral,
hardcoded result if both are unavailable -- this function must never raise,
since the request pipeline depends on it always returning something.
"""

import json
import os
import re

import requests

SYSTEM_PROMPT = """You are the emotional-analysis component of a music-therapy chat app.
Given a short message describing how someone feels, respond with STRICT JSON only,
no markdown fences, no commentary, matching exactly this shape:

{"valence": <float from -1 to 1>, "arousal": <float from -1 to 1>, "reply": "<one short warm sentence>"}

valence: how negative (-1) to positive (1) the feeling is.
arousal: how low-energy/calm (-1) to high-energy/activated (1) the feeling is.
reply: one short, warm, specific sentence acknowledging how they feel. Not a generic
chatbot greeting, not advice, not a question -- just a brief, human acknowledgement,
under 20 words.

Examples:
"I've had a terrible day, I'm exhausted" -> {"valence": -0.6, "arousal": -0.7, "reply": "That sounds like a genuinely draining day -- you've earned some rest."}
"I just got the job!! I can't believe it!" -> {"valence": 0.9, "arousal": 0.8, "reply": "That's wonderful news -- congratulations, you should be thrilled."}
"nothing really happening, just a normal tuesday" -> {"valence": 0.0, "arousal": -0.3, "reply": "Sounds like a quiet, steady kind of day."}
"my flight got cancelled and I'm stuck at the airport" -> {"valence": -0.5, "arousal": 0.5, "reply": "Ugh, that's such a frustrating situation to be stuck in."}
"""

NEUTRAL_FALLBACK = {
    "valence": 0.0,
    "arousal": 0.0,
    "reply": "Thanks for sharing how you're feeling.",
}


def _extract_json(raw: str) -> dict | None:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _validate(data: dict) -> dict | None:
    if not isinstance(data, dict):
        return None
    try:
        valence = float(data["valence"])
        arousal = float(data["arousal"])
        reply = str(data["reply"]).strip()
    except (KeyError, TypeError, ValueError):
        return None
    if not reply:
        return None
    valence = max(-1.0, min(1.0, valence))
    arousal = max(-1.0, min(1.0, arousal))
    return {"valence": valence, "arousal": arousal, "reply": reply}


def _call_groq(user_text: str) -> dict | None:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return None
    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        model = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
        for _attempt in range(2):
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0.4,
                max_tokens=200,
            )
            content = resp.choices[0].message.content
            data = _extract_json(content)
            validated = _validate(data) if data else None
            if validated:
                return validated
        return None
    except Exception as exc:  # noqa: BLE001 - never let an LLM failure crash the request
        print(f"[llm_client] Groq call failed: {exc}")
        return None


def _call_ollama(user_text: str) -> dict | None:
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1")
    try:
        for _attempt in range(2):
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.4},
                },
                timeout=90,  # first call can be slow: Ollama has to load the model into memory
            )
            resp.raise_for_status()
            content = resp.json().get("message", {}).get("content", "")
            data = _extract_json(content)
            validated = _validate(data) if data else None
            if validated:
                return validated
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"[llm_client] Ollama call failed: {exc}")
        return None


def get_emotion_and_reply(user_text: str) -> dict:
    """Returns {"valence": float, "arousal": float, "reply": str}. Never raises."""
    result = _call_groq(user_text)
    if result:
        return result

    result = _call_ollama(user_text)
    if result:
        return result

    print("[llm_client] All LLM providers unavailable, using neutral fallback.")
    return dict(NEUTRAL_FALLBACK)
