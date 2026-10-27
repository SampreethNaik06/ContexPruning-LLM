"""Small Gemini REST client using Python's standard library."""

import json
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .settings import MODEL


def generate(payload: dict, key: str) -> str:
    """Generate an answer without exposing credentials or upstream error bodies."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{quote(MODEL, safe='')}:generateContent"
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers={
        "Content-Type": "application/json", "x-goog-api-key": key,
    }, method="POST")
    try:
        for attempt in range(2):
            try:
                with urlopen(request, timeout=60) as response:
                    result = json.load(response)
                break
            except HTTPError as exc:
                if attempt or exc.code not in (500, 502, 503, 504):
                    raise
                exc.close()
                time.sleep(1)
    except HTTPError as exc:
        messages = {
            400: "Gemini rejected the request. Check the server's API key and model configuration.",
            401: "Gemini authentication failed. Check the server's API key.",
            403: "Gemini access was denied. Check the API key and project permissions.",
            404: "The configured Gemini model is unavailable. Update GEMINI_MODEL on the server.",
            429: "Gemini's quota or rate limit was reached. Check your quota or try again later.",
        }
        raise RuntimeError(messages.get(exc.code, "Gemini is temporarily unavailable. Please try again.")) from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("Could not reach Gemini. Check the connection and try again.") from exc
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Gemini returned an unreadable response. Please try again.") from exc

    candidates = result.get("candidates", [])
    if not candidates:
        raise RuntimeError("Gemini returned no answer or blocked this request. Try rephrasing your question.")
    candidate = candidates[0]
    reason = candidate.get("finishReason")
    if reason not in (None, "STOP", "MAX_TOKENS"):
        raise RuntimeError("Gemini could not complete this answer. Try rephrasing your question.")
    answer = "\n".join(
        part["text"] for part in candidate.get("content", {}).get("parts", [])
        if part.get("text") and not part.get("thought")
    ).strip()
    if not answer:
        raise RuntimeError("Gemini returned an empty answer. Please try again.")
    if reason == "MAX_TOKENS":
        answer += "\n\n[Response reached its length limit. Ask a follow-up to continue.]"
    return answer
