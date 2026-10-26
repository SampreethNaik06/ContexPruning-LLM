"""Claim deduplication and original-source evidence attachment."""

from __future__ import annotations

from typing import Any

import networkx as nx

from .path_traversal import PathRAGTraverser
from .traversal_types import claim_dedup_key, match_nodes, relation_matches


def retrieve_claims(
    state_or_graph: dict[str, Any] | nx.MultiDiGraph,
    entity_anchors: list[str],
    relation_anchors: list[str] | None = None,
    alpha: float = 0.8,
    theta: float = 0.05,
    max_hops: int = 3,
    top_k: int = 15,
    cutoff: float = 0.65,
) -> list[dict[str, Any]]:
    """PathRAG path traversal that retrieves factual claims only and drops duplicate claims.

    Args:
        state_or_graph: Graph state dict (from build_graph) or nx.MultiDiGraph.
        entity_anchors: Query entity keywords/anchors to seed traversal.
        relation_anchors: Optional query relation tokens for flow weighting.
        alpha: Information flow decay rate (PathRAG default 0.8).
        theta: Early-stopping pruning threshold (PathRAG default 0.05).
        max_hops: Maximum path hops between query entity endpoints (default 3).
        top_k: Top K reliable relational paths to extract claims from (default 15).
        cutoff: Fuzzy matching cutoff for matching entity anchors to nodes.

    Returns:
        List of distinct, deduplicated factual claims sorted by path reliability.
    """
    if isinstance(state_or_graph, dict):
        graph = state_or_graph["graph"]
        claims_store = state_or_graph.get("claims", {})
    else:
        graph = state_or_graph
        claims_store = {}

    matches = match_nodes(graph, entity_anchors, cutoff=cutoff)
    if not matches and relation_anchors:
        matches = match_nodes(graph, relation_anchors, cutoff=cutoff)
    matched_node_ids = [m["node_id"] for m in matches]

    if not matched_node_ids:
        return []


    traverser = PathRAGTraverser(
        alpha=alpha,
        theta=theta,
        max_hops=max_hops,
        top_k=top_k,
    )

    paths = traverser.retrieve_paths(graph, matched_node_ids, relation_anchors)
    claims, _ = traverser.extract_and_deduplicate_claims(paths, claims_store, graph=graph)

    # Direct 1-hop fallback: if no multi-hop paths formed (e.g. 1 node or disjoint entities),
    # collect the direct factual claims from the matched nodes directly.
    if not claims:
        seen_facts: dict[str, dict[str, Any]] = {}
        for node_id in matched_node_ids:
            node_claims = graph.nodes[node_id].get("claims", [])
            for c in node_claims:
                cid = c.get("claim_id")
                if not cid:
                    continue
                matches_relation = relation_anchors and relation_matches(
                    c.get("relation", ""), relation_anchors
                )
                rel_bonus = 0.5 if matches_relation else 0.0
                dedup_key = claim_dedup_key(c, cid)

                if dedup_key in seen_facts:
                    existing = seen_facts[dedup_key]
                    existing["path_count"] += 1
                    existing.setdefault("occurrences", []).append({
                        "claim_id": cid,
                        "negated": bool(c.get("negated", False)),
                        "document_id": c.get("document_id"),
                        "sentence_span": c.get("sentence_span"),
                        "supporting_text": c.get("supporting_text"),
                    })
                    if rel_bonus > existing.get("relation_bonus", 0.0):
                        existing["relation_bonus"] = rel_bonus
                else:
                    rec = dict(c)
                    rec["path_reliability"] = 1.0
                    rec["relation_bonus"] = rel_bonus
                    rec["path_count"] = 1
                    rec["best_path_rank"] = 1
                    rec["occurrences"] = [{
                        "claim_id": cid,
                        "negated": bool(c.get("negated", False)),
                        "document_id": c.get("document_id"),
                        "sentence_span": c.get("sentence_span"),
                        "supporting_text": c.get("supporting_text"),
                    }]
                    seen_facts[dedup_key] = rec
        claims = list(seen_facts.values())
        claims.sort(
            key=lambda c: (c.get("relation_bonus", 0.0), float(c.get("confidence", 0.0))),
            reverse=True,
        )



    claims, _ = attach_source_evidence(claims, state_or_graph)
    return claims

def attach_source_evidence(
    claims: list[dict[str, Any]],
    state_or_graph: dict[str, Any] | nx.MultiDiGraph,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach exact stored source chunks to retrieved claims and deduplicate sources.

    Claim traversal may merge repeated facts while retaining their source occurrences.
    This adds the original document/chunk text for each occurrence without copying full
    documents into every claim.
    """
    if not isinstance(state_or_graph, dict):
        return claims, []

    documents = state_or_graph.get("documents", {})
    chunks = state_or_graph.get("chunks", {})
    claims_store = state_or_graph.get("claims", {})
    sources_by_id: dict[str, dict[str, Any]] = {}

    for claim in claims:
        occurrences = claim.get("occurrences") or [{
            "claim_id": claim.get("claim_id"),
            "document_id": claim.get("document_id"),
            "chunk_id": claim.get("chunk_id"),
            "sentence_span": claim.get("sentence_span"),
            "supporting_text": claim.get("supporting_text"),
        }]
        evidence: list[dict[str, Any]] = []
        seen_occurrences: set[tuple[Any, ...]] = set()

        for occurrence in occurrences:
            occurrence_claim = claims_store.get(occurrence.get("claim_id"), {})
            document_id = occurrence.get("document_id") or occurrence_claim.get("document_id")
            chunk_id = occurrence.get("chunk_id") or occurrence_claim.get("chunk_id")
            chunk = chunks.get(chunk_id, {})
            document = documents.get(document_id, {})
            if not chunk and not document:
                continue

            sentence_span = occurrence.get("sentence_span") or occurrence_claim.get("sentence_span")
            occurrence_key = (
                occurrence.get("claim_id"),
                document_id,
                chunk_id,
                tuple(sentence_span or []),
            )
            if occurrence_key in seen_occurrences:
                continue
            seen_occurrences.add(occurrence_key)

            source_text = chunk.get("text") or document.get("text")
            source = {
                "claim_id": occurrence.get("claim_id") or occurrence_claim.get("claim_id"),
                "document_id": document_id,
                "source_name": document.get("source_name"),
                "chunk_id": chunk_id,
                "chunk_index": chunk.get("index"),
                "start_char": chunk.get("start_char"),
                "end_char": chunk.get("end_char"),
                "sentence_span": sentence_span,
                "supporting_text": occurrence.get("supporting_text")
                or occurrence_claim.get("supporting_text"),
                "negated": occurrence_claim.get("negated", claim.get("negated", False)),
                "source_text": source_text,
            }
            evidence.append(source)

            source_key = chunk_id or document_id
            if source_key and source_text is not None:
                sources_by_id.setdefault(source_key, {
                    "document_id": document_id,
                    "source_name": document.get("source_name"),
                    "chunk_id": chunk_id,
                    "chunk_index": chunk.get("index"),
                    "start_char": chunk.get("start_char"),
                    "end_char": chunk.get("end_char"),
                    "text": source_text,
                })

        claim["evidence"] = evidence

    return claims, list(sources_by_id.values())

def retrieve_with_traversal_details(
    state_or_graph: dict[str, Any] | nx.MultiDiGraph,
    entity_anchors: list[str],
    relation_anchors: list[str] | None = None,
    alpha: float = 0.8,
    theta: float = 0.05,
    max_hops: int = 3,
    top_k: int = 15,
    cutoff: float = 0.65,
) -> dict[str, Any]:
    """Retrieve deduplicated claims along with full PathRAG traversal diagnostics.

    Returns a dict with:
        - "claims": deduplicated claims list (claims only)
        - "sources": unique original source chunks referenced by those claims
        - "paths": top-K relational paths with reliability scores S(P)
        - "matched_nodes": matched query entities
        - "stats": traversal statistics (paths_found, duplicates_dropped, etc.)
    """
    if isinstance(state_or_graph, dict):
        graph = state_or_graph["graph"]
        claims_store = state_or_graph.get("claims", {})
    else:
        graph = state_or_graph
        claims_store = {}

    matches = match_nodes(graph, entity_anchors, cutoff=cutoff)
    if not matches and relation_anchors:
        matches = match_nodes(graph, relation_anchors, cutoff=cutoff)
    matched_node_ids = [m["node_id"] for m in matches]


    traverser = PathRAGTraverser(
        alpha=alpha,
        theta=theta,
        max_hops=max_hops,
        top_k=top_k,
    )

    paths = traverser.retrieve_paths(graph, matched_node_ids, relation_anchors)
    claims, duplicates_dropped = traverser.extract_and_deduplicate_claims(
        paths, claims_store, graph=graph
    )

    if not claims:
        seen_facts: dict[str, dict[str, Any]] = {}
        for node_id in matched_node_ids:
            node_claims = graph.nodes[node_id].get("claims", [])
            for c in node_claims:
                cid = c.get("claim_id")
                if not cid:
                    continue
                matches_relation = relation_anchors and relation_matches(
                    c.get("relation", ""), relation_anchors
                )
                rel_bonus = 0.5 if matches_relation else 0.0
                dedup_key = claim_dedup_key(c, cid)

                if dedup_key in seen_facts:
                    duplicates_dropped += 1
                    existing = seen_facts[dedup_key]
                    existing["path_count"] += 1
                    existing.setdefault("occurrences", []).append({
                        "claim_id": cid,
                        "negated": bool(c.get("negated", False)),
                        "document_id": c.get("document_id"),
                        "sentence_span": c.get("sentence_span"),
                        "supporting_text": c.get("supporting_text"),
                    })
                    if rel_bonus > existing.get("relation_bonus", 0.0):
                        existing["relation_bonus"] = rel_bonus
                    if float(c.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                        existing["confidence"] = c.get("confidence")
                        existing["supporting_text"] = c.get("supporting_text")
                        existing["sentence_span"] = c.get("sentence_span")
                else:
                    rec = dict(c)
                    rec["path_reliability"] = 1.0
                    rec["relation_bonus"] = rel_bonus
                    rec["path_count"] = 1
                    rec["best_path_rank"] = 1
                    rec["occurrences"] = [{
                        "claim_id": cid,
                        "negated": bool(c.get("negated", False)),
                        "document_id": c.get("document_id"),
                        "sentence_span": c.get("sentence_span"),
                        "supporting_text": c.get("supporting_text"),
                    }]
                    seen_facts[dedup_key] = rec
        claims = list(seen_facts.values())
        claims.sort(
            key=lambda c: (c.get("relation_bonus", 0.0), float(c.get("confidence", 0.0))),
            reverse=True,
        )




    claims, sources = attach_source_evidence(claims, state_or_graph)

    return {
        "matched_nodes": matches,
        "paths": [p.to_dict() for p in paths],
        "claims": claims,
        "sources": sources,
        "stats": {
            "entity_anchors_count": len(entity_anchors),
            "matched_nodes_count": len(matches),
            "paths_explored": len(paths),
            "claims_extracted": len(claims),
            "duplicates_dropped": duplicates_dropped,
            "alpha": alpha,
            "theta": theta,
            "max_hops": max_hops,
        },
    }
