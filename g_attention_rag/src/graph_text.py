"""Text cleanup, normalization, IDs, and paragraph helpers."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache
from functools import lru_cache

import spacy


def stable_id(prefix: str, *parts: object, length: int = 16) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:length]}"

def clean_source_text(text: str) -> str:
    return re.sub(r"(?<!\\)\*\*(.+?)(?<!\\)\*\*", r"\1", text).strip()

def load_nlp(model_name: str):
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"spaCy model '{model_name}' is not installed. Run:\n"
            f"  python -m spacy download {model_name}"
        ) from exc

def normalized_words(text: str) -> str:
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@lru_cache(maxsize=50_000)
def canonical_form(text: str, nlp) -> str:
    parsed = nlp(normalized_words(text))
    words = [
        (token.lemma_ or token.lower_).casefold()
        for token in parsed
        if not token.is_space and token.lower_ not in {"a", "an", "the"}
    ]
    key = " ".join(words).strip()
    key = re.sub(r"\s+(?:'s|’s)$", "", key)
    key = re.sub(r"^(?:the\s+)?concept\s+of\s+", "", key)
    return key

def paragraph_ranges(text: str) -> list[tuple[int, int]]:
    ranges = [
        (match.start(), match.end())
        for match in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", text, flags=re.S)
    ]
    return ranges or [(0, len(text))]

def paragraph_index_for(offset: int, ranges: list[tuple[int, int]]) -> int:
    for index, (start, end) in enumerate(ranges):
        if start <= offset < end:
            return index
    return max(0, len(ranges) - 1)
