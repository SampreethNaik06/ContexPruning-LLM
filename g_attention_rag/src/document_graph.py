"""Build one source document into a provenance-aware entity graph."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import networkx as nx

from .claim_extraction import extract_sentence_triples, meaningful_phrase, resolve_reference
from .entity_aliases import canonical_display, canonical_key, discover_aliases
from .graph_constants import SCHEMA_VERSION
from .graph_structure import (
    add_entity_node,
    build_traversal_backbone,
    chunk_for_offset,
    ensure_connected,
    graph_diagnostics,
    make_chunks,
    validate_graph_invariants,
)
from .graph_text import (
    clean_source_text,
    load_nlp,
    paragraph_index_for,
    paragraph_ranges,
    stable_id,
)


def build_graph(
    text: str,
    source_name: str,
    model_name: str,
    max_chunk_chars: int,
    add_cooccurrence: bool = False,
    connect_components: bool = True,
    nlp=None,
    corpus_aliases: dict[str, str] | None = None,
    corpus_alias_displays: dict[str, set[str]] | None = None,
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
    document_aliases, document_alias_displays = discover_aliases(cleaned, nlp)
    aliases = document_aliases if corpus_aliases is None else corpus_aliases
    alias_displays = (
        document_alias_displays
        if corpus_alias_displays is None
        else corpus_alias_displays
    )
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
    explicit_topic = next(iter(document_alias_displays), None)

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
                if subject_key in graph else canonical_display(
                    raw_subject, subject_key, nlp, aliases, alias_displays
                )
            )
            object_display = (
                graph.nodes[object_key]["canonical_name"]
                if object_key in graph else canonical_display(
                    raw_object, object_key, nlp, aliases, alias_displays
                )
            )
            subject_spans = triple["subject"].get("spans") or []
            object_spans = triple["object"].get("spans") or []
            subject_span = subject_spans[0] if len(subject_spans) == 1 else None
            object_span = object_spans[0] if len(object_spans) == 1 else None
            mention_common = {
                "document_id": document_id, "chunk_id": chunk_id,
                "sentence_id": sentence_id, "supporting_text": sentence_text,
                "sentence_span": [sentence.start_char, sentence.end_char],
            }
            add_entity_node(
                graph, subject_key, subject_display, triple["subject"].get("ent_type"),
                {
                    **mention_common, "text": raw_subject, "span": subject_span,
                    "spans": subject_spans,
                },
                paragraph_index, alias_displays.get(subject_key, ()),
            )
            add_entity_node(
                graph, object_key, object_display, triple["object"].get("ent_type"),
                {
                    **mention_common, "text": raw_object, "span": object_span,
                    "spans": object_spans,
                },
                paragraph_index, alias_displays.get(object_key, ()),
            )

            claim_id_parts = [
                document_id, sentence_id, subject_key, triple["relation"], object_key,
            ]
            if triple["negated"]:
                claim_id_parts.append("negated")
            claim_id = stable_id("claim", *claim_id_parts)
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
                "subject_span": subject_span, "subject_spans": subject_spans,
                "predicate_span": triple["predicate_span"],
                "object_span": object_span, "object_spans": object_spans,
                "supporting_text": sentence_text,
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
