"""Graph nodes, source chunks, navigation bridges, and structural checks."""

from __future__ import annotations

from typing import Any, Iterable

import networkx as nx
from spacy.tokens import Doc, Span

from .graph_text import stable_id


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
            "span": mention.get("span") or mention.get("sentence_span"),
            "spans": mention.get("spans", []),
            "sentence_span": mention.get("sentence_span"),
            "supporting_text": mention.get("supporting_text", ""),
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
