"""Compatibility facade for phrase, alias, and claim extraction helpers."""

from .claim_extraction import (
    expand_conjuncts,
    extract_sentence_triples,
    meaningful_phrase,
    resolve_reference,
    subjects_for,
    verb_relation,
)
from .entity_aliases import (
    canonical_display,
    canonical_key,
    discover_aliases,
    discover_corpus_aliases,
    entity_covering_token,
    phrase_from_token,
    preferred_display,
)

__all__ = [
    "entity_covering_token", "phrase_from_token", "discover_aliases",
    "discover_corpus_aliases", "canonical_key", "preferred_display",
    "canonical_display", "verb_relation", "expand_conjuncts", "subjects_for",
    "extract_sentence_triples", "meaningful_phrase", "resolve_reference",
]
