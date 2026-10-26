"""Dependency-based subject-relation-object extraction and reference checks."""

from __future__ import annotations

from typing import Any, Iterable

from spacy.tokens import Span, Token

from .entity_aliases import phrase_from_token
from .graph_constants import (
    ALLOWED_PREPOSITIONS,
    DEICTIC_WORDS,
    DESCRIPTOR_HEADS,
    DIRECT_OBJECT_DEPS,
    GENERIC_EXACT,
    GENERIC_HEADS,
    PRONOUN_WORDS,
    SUBJECT_DEPS,
)
from .graph_text import normalized_words


def verb_relation(verb: Token, preposition: Token | None = None) -> str:
    parts = [(verb.lemma_ or verb.lower_).casefold()]
    parts.extend(
        child.lemma_.casefold()
        for child in sorted(verb.children, key=lambda item: item.i)
        if child.dep_ == "prt"
    )
    if preposition is not None:
        parts.append((preposition.lemma_ or preposition.lower_).casefold())
    return "_".join(part for part in parts if part)

def expand_conjuncts(tokens: Iterable[Token]) -> list[Token]:
    expanded: dict[int, Token] = {}
    pending = list(tokens)
    while pending:
        token = pending.pop()
        if token.i in expanded:
            continue
        expanded[token.i] = token
        pending.extend(token.conjuncts)
    return [expanded[index] for index in sorted(expanded)]

def subjects_for(predicate: Token) -> list[Token]:
    subjects = [child for child in predicate.children if child.dep_ in SUBJECT_DEPS]
    if predicate.dep_ == "relcl" and any(
        item.lower_ in {"that", "which", "who"} for item in subjects
    ):
        return [predicate.head]
    if subjects:
        return subjects
    if predicate.dep_ == "acl" and predicate.head.pos_ in {"NOUN", "PROPN"}:
        return [predicate.head]
    if predicate.dep_ in {"advcl", "conj", "xcomp"}:
        inherited = [child for child in predicate.head.children if child.dep_ in SUBJECT_DEPS]
        if inherited:
            return inherited
    return []

def extract_sentence_triples(sentence: Span) -> list[dict[str, Any]]:
    triples: list[dict[str, Any]] = []

    def add(subject_token: Token, relation: str, object_token: Token, method: str,
            confidence: float, predicate: Token, voice: str = "active") -> None:
        subject = phrase_from_token(subject_token)
        obj = phrase_from_token(object_token)
        if subject["text"] and obj["text"]:
            triples.append({
                "subject": subject,
                "relation": relation,
                "object": obj,
                "predicate_span": [predicate.idx, predicate.idx + len(predicate)],
                "method": method,
                "confidence": confidence,
                "voice": voice,
                "negated": any(child.dep_ == "neg" for child in predicate.children),
            })

    for predicate in sentence:
        if predicate.pos_ not in {"AUX", "VERB"}:
            continue
        subjects = expand_conjuncts(subjects_for(predicate))
        if not subjects:
            continue
        agents: list[Token] = []
        for child in predicate.children:
            if child.dep_ == "agent" or (child.dep_ == "prep" and child.lemma_ == "by"):
                agents.extend(item for item in child.children if item.dep_ == "pobj")
        passive = (
            any(subject.dep_ in {"nsubjpass", "csubjpass"} for subject in subjects)
            or bool(agents and predicate.tag_ == "VBN")
        )
        if passive and agents:
            for actor in expand_conjuncts(agents):
                for patient in subjects:
                    add(actor, verb_relation(predicate), patient, "passive_svo", 0.95,
                        predicate, "passive_normalized")
            continue

        objects = expand_conjuncts(
            child for child in predicate.children if child.dep_ in DIRECT_OBJECT_DEPS
        )
        for subject in subjects:
            descriptor_head = (subject.lemma_ or subject.lower_).casefold()
            descriptor_targets = [
                pobj
                for prep in subject.children
                if prep.dep_ == "prep" and prep.lemma_ == "of"
                for pobj in prep.children if pobj.dep_ == "pobj"
            ]
            for obj in objects:
                if (
                    predicate.lemma_ == "be"
                    and descriptor_head in DESCRIPTOR_HEADS
                    and descriptor_targets
                ):
                    for target in descriptor_targets:
                        add(obj, f"be_{descriptor_head}_of", target,
                            "normalized_copular", 0.94, predicate)
                else:
                    relation = "not_be" if (
                        predicate.lemma_ == "be"
                        and any(child.dep_ == "neg" for child in predicate.children)
                    ) else verb_relation(predicate)
                    add(subject, relation, obj, "dependency_svo", 0.96, predicate)

        for prep in predicate.children:
            if prep.dep_ != "prep" or prep.lemma_.casefold() not in ALLOWED_PREPOSITIONS:
                continue
            objects = expand_conjuncts(child for child in prep.children if child.dep_ == "pobj")
            for subject in subjects:
                for obj in objects:
                    add(subject, verb_relation(predicate, prep), obj,
                        "prepositional_svo", 0.86, predicate)

    for complement in sentence:
        copulas = [child for child in complement.children if child.dep_ == "cop"]
        subjects = expand_conjuncts(
            child for child in complement.children if child.dep_ in SUBJECT_DEPS
        )
        if not copulas or not subjects:
            continue
        copula = copulas[0]
        for subject in subjects:
            descriptor_head = (subject.lemma_ or subject.lower_).casefold()
            of_objects = [
                pobj
                for prep in subject.children
                if prep.dep_ == "prep" and prep.lemma_ == "of"
                for pobj in prep.children if pobj.dep_ == "pobj"
            ]
            if descriptor_head in DESCRIPTOR_HEADS and of_objects:
                for target in of_objects:
                    add(complement, f"be_{descriptor_head}_of", target,
                        "normalized_copular", 0.94, copula)
                continue
            if descriptor_head == "one":
                nested = [
                    pobj
                    for prep in subject.subtree if prep.dep_ == "prep" and prep.lemma_ == "of"
                    for pobj in prep.children if pobj.dep_ == "pobj"
                ]
                meaningful = [
                    item for item in nested
                    if item.lemma_.casefold() not in DESCRIPTOR_HEADS
                ]
                if meaningful:
                    add(complement, "be_feature_of", meaningful[-1],
                        "normalized_copular", 0.91, copula)
                continue
            is_negated = any(child.dep_ == "neg" for child in complement.children)
            relation = "be_not" if is_negated else "be"
            for value in expand_conjuncts([complement]):
                add(subject, relation, value, "copular_svo", 0.94, copula)

    unique: dict[tuple[str, str, str, bool], dict[str, Any]] = {}
    for triple in triples:
        key = (
            normalized_words(triple["subject"]["text"]),
            triple["relation"],
            normalized_words(triple["object"]["text"]),
            triple["negated"],
        )
        if key not in unique or triple["confidence"] > unique[key]["confidence"]:
            unique[key] = triple
    return list(unique.values())

def meaningful_phrase(text: str, head: str) -> tuple[bool, str]:
    normalized = normalized_words(text)
    words = normalized.split()
    if not normalized:
        return False, "empty phrase"
    if normalized in PRONOUN_WORDS or normalized in DEICTIC_WORDS:
        return False, "unresolved pronoun"
    if words[0] in DEICTIC_WORDS:
        return False, "unresolved deictic phrase"
    if normalized in GENERIC_EXACT or head in GENERIC_HEADS:
        return False, "generic phrase"
    if len(words) > 10:
        return False, "phrase longer than 10 words"
    if not any(character.isalpha() for character in normalized):
        return False, "phrase has no letters"
    return True, ""

def resolve_reference(
    phrase: dict[str, Any],
    recent_nodes: list[dict[str, str]],
    previous_subject: str | None,
    previous_object: str | None,
) -> str | None:
    normalized = normalized_words(phrase["text"])
    words = normalized.split()
    if normalized in {"it", "he", "she"}:
        return previous_subject or previous_object
    if normalized in {"they", "these", "those"}:
        return previous_subject
    if words and words[0] in DEICTIC_WORDS:
        head = phrase["head"]
        for item in reversed(recent_nodes):
            if head and head in normalized_words(item["display"]).split():
                return item["key"]
        if head in {"discovery", "result"}:
            return previous_object or previous_subject
        return None
    return ""
