"""Build a connected, provenance-aware, entity-anchored SVO graph.

The graph remains completely file based. Factual SVO claims are stored as
MultiDiGraph edges and copied onto both participating entity nodes. Marked
document-context bridges connect otherwise isolated factual components without
pretending those inferred bridges are source claims.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import spacy
from spacy.tokens import Doc, Span, Token


SCHEMA_VERSION = "3.0"
SUBJECT_DEPS = {"nsubj", "csubj", "nsubjpass", "csubjpass"}
DIRECT_OBJECT_DEPS = {"dobj", "obj", "attr", "oprd", "dative"}
DEICTIC_WORDS = {"this", "that", "these", "those"}
PRONOUN_WORDS = {"it", "they", "he", "she", "we", "which", "who", "whom"}
GENERIC_EXACT = {
    "act", "another", "branch", "conclusion", "core", "other", "one",
    "something", "state", "thing", "way", "which", "world",
}
GENERIC_HEADS = {"act", "conclusion", "core", "other", "one", "thing", "way", "which"}
ALLOWED_PREPOSITIONS = {
    "at", "between", "beyond", "by", "during", "for", "from", "in", "into",
    "of", "on", "through", "to", "until", "upon", "with",
}
DESCRIPTOR_HEADS = {"application", "cornerstone", "feature", "field", "form", "type"}
CO_OCCUR_RELATION = "co_occurs"
CONTEXT_RELATIONS = {"covers", "contextualizes", "next_topic"}


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


def entity_covering_token(token: Token) -> Span | None:
    for entity in token.doc.ents:
        if entity.start <= token.i < entity.end:
            return entity
    return None


def phrase_from_token(token: Token, include_of_phrase: bool = True) -> dict[str, Any]:
    """Create a compact entity/concept phrase without swallowing clauses."""
    entity = entity_covering_token(token)
    if entity is not None:
        return {
            "text": entity.text,
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
    text = " ".join(item.text for item in tokens)
    text = re.sub(r"\s+(['’]s)\b", r"\1", text)
    text = re.sub(r"^(?:a|an|the)\s+", "", text.strip(" \t\n,;:.—-"), flags=re.I)
    return {
        "text": text,
        "start": tokens[0].idx,
        "end": tokens[-1].idx + len(tokens[-1]),
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
    for token in tokens:
        expanded[token.i] = token
        expanded.update({conjunct.i: conjunct for conjunct in token.conjuncts})
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
                if predicate.lemma_ == "be" and descriptor_head in DESCRIPTOR_HEADS and descriptor_targets:
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
                meaningful = [item for item in nested if item.lemma_.casefold() not in DESCRIPTOR_HEADS]
                if meaningful:
                    add(complement, "be_feature_of", meaningful[-1],
                        "normalized_copular", 0.91, copula)
                continue
            relation = "be_not" if any(child.dep_ == "neg" for child in complement.children) else "be"
            for value in expand_conjuncts([complement]):
                add(subject, relation, value, "copular_svo", 0.94, copula)

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for triple in triples:
        key = (
            normalized_words(triple["subject"]["text"]),
            triple["relation"],
            normalized_words(triple["object"]["text"]),
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


def make_chunks(doc: Doc, max_chars: int, document_id: str) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    pending: list[Span] = []
    pending_size = 0

    def flush() -> None:
        nonlocal pending, pending_size
        if not pending:
            return
        start, end = pending[0].start_char, pending[-1].end_char
        chunk_id = stable_id("chunk", document_id, start, end)
        chunks[chunk_id] = {
            "chunk_id": chunk_id, "document_id": document_id, "index": len(chunks),
            "start_char": start, "end_char": end, "text": doc.text[start:end],
        }
        pending, pending_size = [], 0

    for sentence in doc.sents:
        size = sentence.end_char - sentence.start_char
        if pending and pending_size + size + 1 > max_chars:
            flush()
        pending.append(sentence)
        pending_size += size + 1
    flush()
    return chunks


def chunk_for_offset(chunks: dict[str, dict[str, Any]], offset: int) -> str:
    for chunk_id, chunk in chunks.items():
        if chunk["start_char"] <= offset < chunk["end_char"]:
            return chunk_id
    raise ValueError(f"No chunk contains document offset {offset}")


def add_entity_node(
    graph: nx.MultiDiGraph,
    key: str,
    display_name: str,
    ent_type: str | None,
    mention: dict[str, Any],
    paragraph_index: int,
    explicit_aliases: Iterable[str] = (),
) -> None:
    if key not in graph:
        graph.add_node(
            key, node_type="entity" if ent_type else "concept",
            canonical_name=display_name, ent_type=ent_type, aliases=set(),
            mentions=[], claims=[], claim_ids=set(), supporting_chunk_ids=set(),
            paragraph_indices=set(),
        )
    node = graph.nodes[key]
    node["aliases"].update({display_name, *explicit_aliases})
    node["mentions"].append(mention)
    node["supporting_chunk_ids"].add(mention["chunk_id"])
    node["paragraph_indices"].add(paragraph_index)
    if node.get("ent_type") is None and ent_type:
        node["ent_type"] = ent_type
        node["node_type"] = "entity"


def ensure_connected(
    graph: nx.MultiDiGraph,
    topic_key: str,
    document_id: str,
    paragraph_anchors: dict[int, str],
) -> list[dict[str, Any]]:
    """Connect components through local paragraph anchors, not a topic star."""
    bridges: list[dict[str, Any]] = []
    if topic_key not in graph or graph.number_of_nodes() < 2:
        return bridges

    def add_bridge(source: str, target: str, relation: str) -> None:
        if source == target or nx.has_path(graph.to_undirected(), source, target):
            return
        mentions = graph.nodes[target].get("mentions", [])
        mention = mentions[0] if mentions else {}
        edge_id = stable_id("context", document_id, source, relation, target)
        data = {
            "relation": relation, "claim_id": None, "confidence": 0.55,
            "weight": 0.15, "negated": False, "inferred": True,
            "edge_type": "document_context", "document_id": document_id,
            "chunk_id": mention.get("chunk_id"), "sentence_id": mention.get("sentence_id"),
            "span": mention.get("span"), "supporting_text": mention.get("supporting_text", ""),
        }
        graph.add_edge(source, target, key=edge_id, **data)
        bridges.append({"source": source, "target": target, "key": edge_id, **data})

    paragraph_ids = sorted({
        index
        for _, data in graph.nodes(data=True)
        for index in data.get("paragraph_indices", set())
    })
    local_anchors: dict[int, str] = {}
    for paragraph_index in paragraph_ids:
        candidates = [
            node for node, data in graph.nodes(data=True)
            if paragraph_index in data.get("paragraph_indices", set())
        ]
        non_topic = [node for node in candidates if node != topic_key]
        pool = non_topic if paragraph_index > 0 and non_topic else candidates
        if pool:
            local_anchors[paragraph_index] = max(
                pool,
                key=lambda node: (
                    len(graph.nodes[node].get("claims", [])),
                    graph.degree(node),
                    graph.nodes[node].get("ent_type") is not None,
                ),
            )
        elif paragraph_index in paragraph_anchors:
            local_anchors[paragraph_index] = paragraph_anchors[paragraph_index]

    # Form a document-order spine. This distributes traversal across local
    # paragraph topics rather than attaching every section to the root topic.
    previous_anchor = topic_key
    for paragraph_index in paragraph_ids:
        anchor = local_anchors.get(paragraph_index)
        if anchor not in graph or anchor == previous_anchor:
            continue
        add_bridge(previous_anchor, anchor, "next_topic")
        previous_anchor = anchor

    while True:
        components = list(nx.weakly_connected_components(graph))
        if len(components) <= 1:
            break
        main = next(component for component in components if topic_key in component)
        for component in [item for item in components if item is not main]:
            representative = max(
                component,
                key=lambda node: (
                    graph.degree(node), len(graph.nodes[node].get("claims", [])),
                    -min(graph.nodes[node].get("paragraph_indices", {9999})),
                ),
            )
            paragraph_index = min(graph.nodes[representative].get("paragraph_indices", {0}))
            local_anchor = local_anchors.get(paragraph_index, previous_anchor)
            if local_anchor in component:
                outside_anchors = [
                    anchor for _, anchor in sorted(local_anchors.items())
                    if anchor not in component
                ]
                local_anchor = outside_anchors[-1] if outside_anchors else topic_key
            add_bridge(local_anchor, representative, "contextualizes")
    return bridges


def build_traversal_backbone(
    graph: nx.MultiDiGraph,
    topic_key: str,
) -> nx.MultiDiGraph:
    """Create a maximum-confidence spanning backbone without deleting evidence."""
    candidates = nx.Graph()
    candidates.add_nodes_from(graph.nodes)
    for source, target, key, data in graph.edges(keys=True, data=True):
        if source == target:
            continue
        edge_type = data.get("edge_type", "factual_claim")
        base_priority = {"factual_claim": 100.0, "document_context": 10.0}.get(edge_type, 1.0)
        priority = base_priority + float(data.get("confidence", 0.0))
        current = candidates.get_edge_data(source, target)
        if current is None or priority > current["priority"]:
            candidates.add_edge(
                source, target, priority=priority,
                original=(source, target, key, dict(data)),
            )

    tree = nx.maximum_spanning_tree(candidates, weight="priority")
    backbone = nx.MultiDiGraph(
        name=f"Traversal backbone for {graph.graph.get('name', 'knowledge graph')}",
        schema_version=graph.graph.get("schema_version"),
        document_id=graph.graph.get("document_id"),
        topic_node=topic_key,
        graph_role="traversal_backbone",
    )
    backbone.add_nodes_from((node, dict(data)) for node, data in graph.nodes(data=True))
    for _, _, tree_data in tree.edges(data=True):
        source, target, key, data = tree_data["original"]
        backbone.add_edge(source, target, key=key, **data)

    if backbone.number_of_nodes() > 1 and nx.is_weakly_connected(graph):
        if nx.number_weakly_connected_components(backbone) != 1:
            raise ValueError("Traversal backbone is unexpectedly disconnected")
        if backbone.number_of_edges() != backbone.number_of_nodes() - 1:
            raise ValueError("Traversal backbone is not a spanning tree")
    return backbone


def validate_graph_invariants(
    graph: nx.MultiDiGraph,
    claims: dict[str, dict[str, Any]],
    source_text: str,
    require_connected: bool,
) -> None:
    """Fail the build when structural or provenance guarantees are broken."""
    errors: list[str] = []
    if require_connected and nx.number_weakly_connected_components(graph) != 1:
        errors.append("graph is not a single weakly connected component")
    for claim_id, claim in claims.items():
        subject, object_ = claim["subject_id"], claim["object_id"]
        if subject not in graph or object_ not in graph:
            errors.append(f"{claim_id}: references a missing node")
            continue
        if not graph.has_edge(subject, object_, key=claim_id):
            errors.append(f"{claim_id}: claim edge is missing")
        start, end = claim["sentence_span"]
        if source_text[start:end].strip() != claim["supporting_text"].strip():
            errors.append(f"{claim_id}: sentence provenance does not round trip")
        for node_id in (subject, object_):
            if claim_id not in graph.nodes[node_id].get("claim_ids", set()):
                errors.append(f"{claim_id}: missing from node {node_id}")
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:20])
        raise ValueError(f"Knowledge-graph validation failed:\n{preview}")


def graph_diagnostics(
    graph: nx.MultiDiGraph, traversal_graph: nx.MultiDiGraph,
    claims: dict[str, Any], context_bridges: list, rejected_triples: list,
) -> dict[str, Any]:
    """Summarize a graph after a document build or collection merge."""
    return {
        "node_count": graph.number_of_nodes(),
        "factual_claim_count": len(claims),
        "context_bridge_count": len(context_bridges),
        "component_count": nx.number_weakly_connected_components(graph),
        "traversal_node_count": traversal_graph.number_of_nodes(),
        "traversal_edge_count": traversal_graph.number_of_edges(),
        "traversal_component_count": nx.number_weakly_connected_components(traversal_graph),
        "rejected_triple_count": len(rejected_triples),
        "topic_node": graph.graph["topic_node"],
    }


def build_graph(
    text: str,
    source_name: str,
    model_name: str,
    max_chunk_chars: int,
    add_cooccurrence: bool = False,
    connect_components: bool = True,
    nlp=None,
) -> dict[str, Any]:
    del add_cooccurrence  # Retained only for command/API compatibility.
    if max_chunk_chars < 100:
        raise ValueError("max_chunk_chars must be at least 100")
    if not text.strip():
        raise ValueError("Input document is empty")
    nlp = load_nlp(model_name) if nlp is None else nlp
    cleaned = clean_source_text(text)
    doc = nlp(cleaned)
    document_id = stable_id("doc", source_name, cleaned)
    ranges = paragraph_ranges(cleaned)
    aliases, alias_displays = discover_aliases(cleaned, nlp)
    chunks = make_chunks(doc, max_chunk_chars, document_id)
    graph = nx.MultiDiGraph(
        name=f"Entity-anchored SVO graph for {source_name}",
        schema_version=SCHEMA_VERSION, document_id=document_id,
    )

    claims: dict[str, dict[str, Any]] = {}
    rejected: list[dict[str, Any]] = []
    recent_nodes: list[dict[str, str]] = []
    previous_subject: str | None = None
    previous_object: str | None = None
    paragraph_anchors: dict[int, str] = {}
    subject_frequency: Counter[str] = Counter()
    explicit_topic = next(iter(alias_displays), None)

    for sentence_index, sentence in enumerate(doc.sents):
        paragraph_index = paragraph_index_for(sentence.start_char, ranges)
        chunk_id = chunk_for_offset(chunks, sentence.start_char)
        sentence_id = stable_id("sent", document_id, sentence.start_char, sentence.end_char)
        sentence_text = sentence.text.strip()

        for triple in extract_sentence_triples(sentence):
            subject_reference = resolve_reference(
                triple["subject"], recent_nodes, previous_subject, previous_object
            )
            object_reference = resolve_reference(
                triple["object"], recent_nodes, previous_subject, previous_object
            )
            raw_subject, raw_object = triple["subject"]["text"], triple["object"]["text"]
            subject_key = subject_reference or canonical_key(raw_subject, nlp, aliases)
            object_key = object_reference or canonical_key(raw_object, nlp, aliases)
            subject_ok, subject_reason = meaningful_phrase(
                raw_subject if subject_reference == "" else subject_key,
                triple["subject"]["head"],
            )
            object_ok, object_reason = meaningful_phrase(
                raw_object if object_reference == "" else object_key,
                triple["object"]["head"],
            )

            rejection_reason = ""
            if subject_reference is None or object_reference is None:
                rejection_reason = "unresolved reference"
            elif not subject_ok:
                rejection_reason = f"invalid subject: {subject_reason}"
            elif not object_ok:
                rejection_reason = f"invalid object: {object_reason}"
            elif subject_key == object_key:
                rejection_reason = "self-referential triple"
            elif triple["confidence"] < 0.80:
                rejection_reason = "confidence below threshold"
            if rejection_reason:
                rejected.append({
                    "sentence": sentence_text, "subject": raw_subject,
                    "relation": triple["relation"], "object": raw_object,
                    "reason": rejection_reason,
                })
                continue

            subject_display = (
                graph.nodes[subject_key]["canonical_name"]
                if subject_key in graph else preferred_display(raw_subject)
            )
            object_display = (
                graph.nodes[object_key]["canonical_name"]
                if object_key in graph else preferred_display(raw_object)
            )
            subject_span = [triple["subject"]["start"], triple["subject"]["end"]]
            object_span = [triple["object"]["start"], triple["object"]["end"]]
            mention_common = {
                "document_id": document_id, "chunk_id": chunk_id,
                "sentence_id": sentence_id, "supporting_text": sentence_text,
            }
            add_entity_node(
                graph, subject_key, subject_display, triple["subject"].get("ent_type"),
                {**mention_common, "text": raw_subject, "span": subject_span},
                paragraph_index, alias_displays.get(subject_key, ()),
            )
            add_entity_node(
                graph, object_key, object_display, triple["object"].get("ent_type"),
                {**mention_common, "text": raw_object, "span": object_span},
                paragraph_index, alias_displays.get(object_key, ()),
            )

            claim_id = stable_id(
                "claim", document_id, sentence_id, subject_key, triple["relation"], object_key
            )
            claim = {
                "claim_id": claim_id, "subject_id": subject_key,
                "subject": graph.nodes[subject_key]["canonical_name"],
                "relation": triple["relation"], "object_id": object_key,
                "object": graph.nodes[object_key]["canonical_name"],
                "negated": triple["negated"], "voice": triple["voice"],
                "confidence": triple["confidence"], "extraction_method": triple["method"],
                "document_id": document_id, "chunk_id": chunk_id,
                "sentence_id": sentence_id, "sentence_index": sentence_index,
                "paragraph_index": paragraph_index,
                "sentence_span": [sentence.start_char, sentence.end_char],
                "subject_span": subject_span, "predicate_span": triple["predicate_span"],
                "object_span": object_span, "supporting_text": sentence_text,
                "fact": (
                    f"{graph.nodes[subject_key]['canonical_name']} "
                    f"{triple['relation'].replace('_', ' ')} "
                    f"{graph.nodes[object_key]['canonical_name']}"
                ),
            }
            claims[claim_id] = claim
            graph.add_edge(
                subject_key, object_key, key=claim_id, relation=triple["relation"],
                claim_id=claim_id, confidence=triple["confidence"],
                weight=triple["confidence"], negated=triple["negated"],
                inferred=False, edge_type="factual_claim", document_id=document_id,
                chunk_id=chunk_id, sentence_id=sentence_id,
                span=[sentence.start_char, sentence.end_char], supporting_text=sentence_text,
            )
            for node_key, role in ((subject_key, "subject"), (object_key, "object")):
                node_claim = dict(claim)
                node_claim["entity_role"] = role
                graph.nodes[node_key]["claims"].append(node_claim)
                graph.nodes[node_key]["claim_ids"].add(claim_id)

            subject_frequency[subject_key] += 1
            paragraph_anchors.setdefault(paragraph_index, subject_key)
            previous_subject, previous_object = subject_key, object_key
            recent_nodes.extend([
                {"key": subject_key, "display": graph.nodes[subject_key]["canonical_name"]},
                {"key": object_key, "display": graph.nodes[object_key]["canonical_name"]},
            ])
            recent_nodes = recent_nodes[-24:]

    if graph.number_of_nodes() == 0:
        raise ValueError("No valid factual triples were extracted from the document.")

    topic_key = explicit_topic if explicit_topic in graph else max(
        graph.nodes, key=lambda node: (subject_frequency[node], graph.degree(node))
    )
    graph.graph["topic_node"] = topic_key
    graph.nodes[topic_key]["is_document_topic"] = True
    context_bridges = (
        ensure_connected(graph, topic_key, document_id, paragraph_anchors)
        if connect_components else []
    )
    validate_graph_invariants(graph, claims, cleaned, connect_components)
    traversal_graph = build_traversal_backbone(graph, topic_key)
    diagnostics = graph_diagnostics(graph, traversal_graph, claims, context_bridges, rejected)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_name": source_name, "document_id": document_id,
        "source_text": cleaned,
        "documents": {document_id: {"source_name": source_name, "text": cleaned}},
        "model_name": model_name, "topic_node": topic_key,
        "explicit_aliases": aliases, "graph": graph,
        "traversal_graph": traversal_graph, "chunks": chunks,
        "claims": claims, "context_bridges": context_bridges,
        "rejected_triples": rejected, "diagnostics": diagnostics,
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(json_safe(item) for item in value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def graph_to_dict(state: dict[str, Any]) -> dict[str, Any]:
    graph: nx.MultiDiGraph = state["graph"]
    traversal_graph: nx.MultiDiGraph = state["traversal_graph"]
    payload = {
        key: value for key, value in state.items()
        if key not in {"graph", "traversal_graph"}
    }
    payload["nodes"] = [
        {"id": node_id, **json_safe(attributes)}
        for node_id, attributes in graph.nodes(data=True)
    ]
    payload["edges"] = [
        {"source": source, "target": target, "key": key, **json_safe(attributes)}
        for source, target, key, attributes in graph.edges(keys=True, data=True)
    ]
    payload["traversal_nodes"] = [
        {"id": node_id, **json_safe(attributes)}
        for node_id, attributes in traversal_graph.nodes(data=True)
    ]
    payload["traversal_edges"] = [
        {"source": source, "target": target, "key": key, **json_safe(attributes)}
        for source, target, key, attributes in traversal_graph.edges(keys=True, data=True)
    ]
    return json_safe(payload)


def export_json(state: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(graph_to_dict(state), indent=2, ensure_ascii=False), encoding="utf-8")


class KnowledgeGraphBuilder:
    """Project-facing builder retaining the full Module 2 serialization contract."""

    def __init__(self, spacy_model: str = "en_core_web_sm", max_chunk_chars: int = 1200):
        self.spacy_model = spacy_model
        self.max_chunk_chars = max_chunk_chars
        self.nlp = None
        self.state = None
        self.graph = nx.MultiDiGraph()

    def add_chunk(self, chunk_id: str, text: str) -> None:
        """Index a source independently so references never cross documents."""
        if self.nlp is None:
            self.nlp = load_nlp(self.spacy_model)
        incoming = build_graph(text, chunk_id, self.spacy_model, self.max_chunk_chars, nlp=self.nlp)
        if self.state is None:
            self.state = incoming
            self.graph = incoming["graph"]
            return
        if incoming["document_id"] in self.state["documents"]:
            return
        for node, attributes in incoming["graph"].nodes(data=True):
            if node not in self.graph:
                self.graph.add_node(node, **attributes)
                continue
            existing = self.graph.nodes[node]
            for field in ("aliases", "claim_ids", "supporting_chunk_ids", "paragraph_indices"):
                existing[field].update(attributes[field])
            for field in ("mentions", "claims"):
                existing[field].extend(attributes[field])
            if not existing.get("ent_type") and attributes.get("ent_type"):
                existing["ent_type"] = attributes["ent_type"]
                existing["node_type"] = "entity"
        self.graph.add_edges_from(incoming["graph"].edges(keys=True, data=True))
        for field in ("documents", "chunks", "claims", "explicit_aliases"):
            self.state[field].update(incoming[field])
        for field in ("context_bridges", "rejected_triples"):
            self.state[field].extend(incoming[field])
        # Cross-document links remain explicitly inferred navigation aids.
        self.state["context_bridges"].extend(ensure_connected(
            self.graph, self.state["topic_node"], "corpus", {}
        ))
        self.state["traversal_graph"] = build_traversal_backbone(self.graph, self.state["topic_node"])
        self.state.pop("source_text", None)
        self.state["source_name"] = "document collection"
        self.state["diagnostics"].update(graph_diagnostics(
            self.graph, self.state["traversal_graph"], self.state["claims"],
            self.state["context_bridges"], self.state["rejected_triples"],
        ))

    def save(self, filepath: str = "data/knowledge_graph.pkl") -> None:
        if self.state is None:
            raise ValueError("No documents have been indexed")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str = "data/knowledge_graph.pkl", spacy_model: str = "en_core_web_sm"):
        builder = cls(spacy_model)
        with Path(filepath).open("rb") as handle:
            builder.state = pickle.load(handle)
        if builder.state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Legacy graph cache: rebuild it with the Module 2 pipeline")
        builder.graph = builder.state["graph"]
        return builder

    def stats(self) -> dict[str, int]:
        return {"num_nodes": self.graph.number_of_nodes(), "num_edges": self.graph.number_of_edges()}

    def to_dict(self) -> dict[str, Any]:
        if self.state is None:
            raise ValueError("No documents have been indexed")
        return {**graph_to_dict(self.state), "stats": self.stats()}


def build_graph_from_documents(
    documents: dict[str, str],
    output_path: str | None = "data/knowledge_graph.pkl",
    *,
    spacy_model: str = "en_core_web_sm",
    max_chunk_chars: int = 1200,
) -> KnowledgeGraphBuilder:
    """Index documents with independent source offsets into one connected graph."""
    if not documents:
        raise ValueError("At least one document is required")
    builder = KnowledgeGraphBuilder(spacy_model, max_chunk_chars)
    for name, text in documents.items():
        builder.add_chunk(name, text)
    if output_path is not None:
        builder.save(output_path)
    return builder
