"""Mention phrase construction and local or corpus-wide alias resolution."""

from __future__ import annotations

import re
from typing import Any, Iterable

from spacy.tokens import Span, Token

from .graph_text import canonical_form


def entity_covering_token(token: Token) -> Span | None:
    """Return a token's full named-entity span without scanning every entity."""
    if token.ent_iob_ == "O":
        return None

    start = token.i
    while start > 0 and token.doc[start].ent_iob_ == "I":
        start -= 1

    end = token.i + 1
    while end < len(token.doc) and token.doc[end].ent_iob_ == "I":
        end += 1

    return token.doc[start:end]

def phrase_from_token(token: Token, include_of_phrase: bool = True) -> dict[str, Any]:
    """Create a compact entity/concept phrase without swallowing clauses."""
    entity = entity_covering_token(token)
    if entity is not None:
        spans = [[entity.start_char, entity.end_char]]
        return {
            "text": entity.text,
            "spans": spans,
            "span": spans[0],
            "start": entity.start_char,
            "end": entity.end_char,
            "head": (token.lemma_ or token.lower_).casefold(),
            "ent_type": entity.label_,
        }

    allowed = {"amod", "case", "compound", "det", "nummod", "poss", "quantmod"}
    selected = {token.i}
    for child in token.children:
        if child.dep_ in allowed:
            selected.add(child.i)
            selected.update(
                grandchild.i for grandchild in child.children
                if grandchild.dep_ in {"amod", "compound", "nummod", "poss"}
            )

    if include_of_phrase:
        for prep in token.children:
            if prep.dep_ == "prep" and prep.lemma_ == "of":
                objects = [child for child in prep.children if child.dep_ == "pobj"]
                if objects:
                    selected.add(prep.i)
                    pobj = objects[0]
                    selected.add(pobj.i)
                    selected.update(
                        child.i for child in pobj.children
                        if child.dep_ in {"amod", "compound", "nummod", "poss"}
                    )
                break

    tokens = [token.doc[index] for index in sorted(selected)]
    if tokens and tokens[0].lower_ in {"a", "an", "the"}:
        tokens = tokens[1:]
    if not tokens:
        tokens = [token]

    text = " ".join(item.text for item in tokens)
    text = re.sub(r"\s+(['’]s)\b", r"\1", text)
    text = re.sub(r"^(?:a|an|the)\s+", "", text.strip(" \t\n,;:.—-"), flags=re.I)
    spans: list[list[int]] = []
    previous_token_index: int | None = None
    for item in tokens:
        start, end = item.idx, item.idx + len(item)
        if spans and previous_token_index == item.i - 1:
            spans[-1][1] = end
        else:
            spans.append([start, end])
        previous_token_index = item.i
    single_span = spans[0] if len(spans) == 1 else None
    return {
        "text": text,
        "spans": spans,
        "span": single_span,
        "start": single_span[0] if single_span else None,
        "end": single_span[1] if single_span else None,
        "head": (token.lemma_ or token.lower_).casefold(),
        "ent_type": None,
    }

def discover_aliases(text: str, nlp) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Discover explicit aliases, including `also known as` and acronyms."""
    mapping: dict[str, str] = {}
    displays: dict[str, set[str]] = {}
    known_as = re.compile(
        r"(?:^|(?<=[.!?])\s+)([^,.;!?]{2,70}),\s+also known as\s+([^,.;!?]{2,70}),",
        flags=re.I,
    )
    for match in known_as.finditer(text):
        primary_text, alias_text = match.group(1).strip(), match.group(2).strip()
        primary, alias = canonical_form(primary_text, nlp), canonical_form(alias_text, nlp)
        if primary and alias:
            mapping[alias] = primary
            displays.setdefault(primary, set()).update({primary_text, alias_text})

    parenthetical = re.compile(r"\b([A-Za-z][A-Za-z -]{3,60})\s*\(([A-Z][A-Z0-9-]{1,9})\)")
    for match in parenthetical.finditer(text):
        primary_text, alias_text = match.group(1).strip(), match.group(2).strip()
        primary_text = re.sub(r"^(?:and|or|such as)\s+", "", primary_text, flags=re.I)
        primary, alias = canonical_form(primary_text, nlp), canonical_form(alias_text, nlp)
        if primary and alias:
            mapping[alias] = primary
            displays.setdefault(primary, set()).update({primary_text, alias_text})
    return mapping, displays

def discover_corpus_aliases(
    texts: Iterable[str], nlp
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Build a corpus-wide alias map, excluding aliases with conflicting targets."""
    alias_targets: dict[str, set[str]] = {}
    target_displays: dict[str, set[str]] = {}
    for text in texts:
        aliases, displays = discover_aliases(text, nlp)
        for alias, target in aliases.items():
            alias_targets.setdefault(alias, set()).add(target)
        for target, names in displays.items():
            target_displays.setdefault(target, set()).update(names)

    ambiguous_aliases = {
        alias for alias, targets in alias_targets.items() if len(targets) > 1
    }
    corpus_aliases = {
        alias: next(iter(targets))
        for alias, targets in alias_targets.items()
        if len(targets) == 1
    }
    corpus_displays = {
        target: {
            name for name in names
            if canonical_form(name, nlp) not in ambiguous_aliases
        }
        for target, names in target_displays.items()
    }
    return corpus_aliases, corpus_displays

def canonical_key(text: str, nlp, aliases: dict[str, str]) -> str:
    key = canonical_form(text, nlp)
    seen: set[str] = set()
    while key in aliases and key not in seen:
        seen.add(key)
        key = aliases[key]
    return key

def preferred_display(text: str) -> str:
    display = re.sub(r"^(?:the\s+)?concept\s+of\s+", "", text.strip(), flags=re.I)
    display = re.sub(r"(?:'s|’s)$", "", display).strip()
    return display

def canonical_display(
    text: str,
    key: str,
    nlp,
    aliases: dict[str, str],
    alias_displays: dict[str, set[str]],
) -> str:
    """Prefer a corpus-wide full-name display over an acronym encountered first."""
    candidates = [
        name for name in alias_displays.get(key, ())
        if canonical_key(name, nlp, aliases) == key
    ]
    if candidates:
        return preferred_display(max(candidates, key=lambda name: (len(name), name.casefold())))
    return preferred_display(text)
