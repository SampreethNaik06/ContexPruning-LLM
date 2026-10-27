"""Backend configuration; credentials never enter browser assets."""

import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _load_env_file() -> None:
    """Load simple KEY=value settings, keeping existing environment values."""
    try:
        lines = (PROJECT_DIR / ".env").read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or not name:
            continue
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        os.environ.setdefault(name, value)


_load_env_file()

STATIC_DIR = Path(__file__).resolve().parent / "static"
GRAPH_PATH = PROJECT_DIR / "data" / "knowledge_graph.pkl"
MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_QUERY_CHARS = 4000
MAX_HISTORY_MESSAGES = 12


def api_key() -> str:
    """Read the Gemini key loaded from .env or the process environment."""
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Gemini is not configured. Set GEMINI_API_KEY in .env or the server environment and restart it.")
    return key
