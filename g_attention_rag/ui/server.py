"""Local HTTP routes for the chat UI. Run: python -m ui.server."""

import argparse
import json
import logging
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import gemini
from .prompt import build_payload
from .retrieval import Retriever
from .settings import MAX_HISTORY_MESSAGES, MAX_QUERY_CHARS, STATIC_DIR, api_key


def validate_request(data: object) -> tuple[str, list[dict]]:
    """Reject invalid questions and limit client-provided conversation context."""
    if not isinstance(data, dict) or not isinstance(data.get("query"), str):
        raise ValueError("Enter a question to continue.")
    query = data["query"].strip()
    if not query or len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Your question must contain 1–{MAX_QUERY_CHARS} characters.")
    history = data.get("history", [])
    if not isinstance(history, list) or len(history) > MAX_HISTORY_MESSAGES:
        raise ValueError("Invalid conversation history. Start a new conversation.")
    if len(history) % 2:
        raise ValueError("Conversation history must contain completed turns.")
    for index, item in enumerate(history):
        expected = "user" if index % 2 == 0 else "assistant"
        if (not isinstance(item, dict) or item.get("role") != expected
                or not isinstance(item.get("text"), str)
                or not 1 <= len(item["text"].strip()) <= 20000):
            raise ValueError("Invalid conversation history. Start a new conversation.")
    return query, history


class ChatHandler(BaseHTTPRequestHandler):
    """Serve only the three public assets and the query endpoint."""

    retriever = Retriever()

    def _respond(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, status: int, data: dict) -> None:
        self._respond(status, json.dumps(data).encode("utf-8"), "application/json; charset=utf-8")

    def _local_request(self) -> bool:
        port = self.server.server_port
        hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        return (self.headers.get("Host") in hosts
                and (origin is None or origin in {f"http://{host}" for host in hosts})
                and self.headers.get("Sec-Fetch-Site") != "cross-site")

    def do_GET(self) -> None:
        if not self._local_request():
            self._json(403, {"error": "Use the local application address."})
            return
        paths = {"/": "index.html", "/styles.css": "styles.css", "/app.js": "app.js"}
        name = paths.get(urlsplit(self.path).path)
        if name is None:
            self._json(404, {"error": "Not found."})
            return
        content_type = mimetypes.guess_type(name)[0] or "text/plain"
        self._respond(200, (STATIC_DIR / name).read_bytes(), f"{content_type}; charset=utf-8")

    def do_POST(self) -> None:
        if not self._local_request():
            self._json(403, {"error": "Use the local application address."})
            return
        if self.path != "/api/chat":
            self._json(404, {"error": "Not found."})
            return
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            self._json(415, {"error": "Send the question as JSON."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 300000:
                raise ValueError("The request is empty or too large.")
            query, history = validate_request(json.loads(self.rfile.read(length)))
        except (ValueError, UnicodeError) as exc:
            self._json(400, {"error": str(exc) if not isinstance(exc, json.JSONDecodeError) else "Invalid JSON request."})
            return
        try:
            key = api_key()
            retrieval = self.retriever.retrieve(query, history)
            answer = gemini.generate(build_payload(query, history, retrieval), key)
            self._json(200, {"answer": answer})
        except RuntimeError as exc:
            self._json(503, {"error": str(exc)})
        except Exception:
            logging.exception("Chat request failed")
            self._json(500, {"error": "The answer could not be prepared. Check the server console and try again."})


def main() -> None:
    """Start the loopback-only development application."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Port must be between 1 and 65535.")
    with ThreadingHTTPServer(("127.0.0.1", args.port), ChatHandler) as server:
        print(f"Lumen is ready at http://127.0.0.1:{args.port}", flush=True)
        try:
            api_key()
        except RuntimeError as exc:
            print(str(exc), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nLumen stopped.")


if __name__ == "__main__":
    main()
