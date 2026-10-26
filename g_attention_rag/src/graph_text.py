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

def heading_ranges(text: str) -> list[tuple[int, int]]:
    """Find short standalone headings that can be joined to prose by spaCy."""
    ignored_title_words = {
        "a", "an", "and", "as", "at", "by", "for", "from", "in", "into",
        "of", "on", "or", "the", "to", "via", "with",
    }
    ranges: list[tuple[int, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        stripped = content.strip()
        leading = len(content) - len(content.lstrip())
        start, end = offset + leading, offset + len(content.rstrip())
        words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", stripped)
        if (
            stripped
            and len(stripped) <= 140
            and 0 < len(words) <= 14
            and not re.search(r"[.!?]\s*$", stripped)
        ):
            content_words = [
                word for word in words if word.casefold() not in ignored_title_words
            ]
            capitalized = sum(word[0].isupper() for word in content_words)
            letters = [char for char in stripped if char.isalpha()]
            uppercase_ratio = (
                sum(char.isupper() for char in letters) / len(letters)
                if letters else 0.0
            )
            numbered = bool(re.match(r"^\s*\d+(?:\.\d+)*[.)]?\s+", stripped))
            markdown_heading = stripped.startswith("#")
            title_case = bool(content_words) and capitalized / len(content_words) >= 0.7
            all_caps = len(letters) >= 3 and uppercase_ratio >= 0.85
            if numbered or markdown_heading or title_case or all_caps:
                ranges.append((start, end))
        offset += len(line)
    return ranges

def mask_heading_text(
    text: str,
    ranges: list[tuple[int, int]] | None = None,
) -> str:
    """Replace standalone headings with spaces while preserving source offsets."""
    chars = list(text)
    for start, end in (heading_ranges(text) if ranges is None else ranges):
        for index in range(start, end):
            if chars[index] not in "\r\n":
                chars[index] = " "
    return "".join(chars)

def paragraph_index_for(offset: int, ranges: list[tuple[int, int]]) -> int:
    for index, (start, end) in enumerate(ranges):
        if start <= offset < end:
            return index
    return max(0, len(ranges) - 1)
