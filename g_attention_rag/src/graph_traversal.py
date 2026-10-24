"""Module 3: PathRAG-inspired Bounded Path Traverser & Claim Extraction.

Based on:
'PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths'
(arXiv:2502.14902, Chen et al., 2025).

Key Capabilities:
1. Seed Anchor Matching: Matches query entity/relation anchors against graph nodes and aliases.
2. Distance-Aware Flow-Based Traversal: Propagates resource flow S(v) with decay rate (alpha)
   and early-stopping pruning (theta) to eliminate noisy branches.
3. Path Reliability Scoring: Evaluates path reliability S(P) from average edge flow and confidence.
4. Factual Claim Extraction: Extracts claims from path edges while excluding non-factual context bridges.
5. Deduplication: Drops duplicate claims across multi-hop paths, tracking peak path reliability,
   path count, and provenance spans.
"""

from __future__ import annotations

import difflib
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import networkx as nx

# Relations reserved for navigation bridges that should never be returned as factual claims
CONTEXT_RELATIONS = {"covers", "contextualizes", "next_topic"}


def normalize_token(text: str) -> str:
    """Normalize text for consistent lexical matching."""
    text = text.casefold().replace("’", "'")
    text = re.sub(r"[^\w\s'-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    """Return set of normalized non-trivial tokens."""
    return {w for w in normalize_token(text).split() if len(w) > 1}


def match_nodes(
    graph: nx.MultiDiGraph,
    entity_anchors: list[str],
    cutoff: float = 0.65,
) -> list[dict[str, Any]]:
    """Match query entity anchors to graph nodes using alias and fuzzy matching.

    Returns a list of match dicts:
        {"anchor": anchor, "node_id": node_id, "score": float, "canonical_name": str}
    """
    matches: list[dict[str, Any]] = []
    seen_nodes: set[str] = set()

    for anchor in entity_anchors:
        wanted = normalize_token(anchor)
        if not wanted:
            continue
        wanted_tokens = token_set(wanted)
        best_node, best_score = None, 0.0

        for node_id, data in graph.nodes(data=True):
            canonical = data.get("canonical_name", str(node_id))
            candidates = [str(node_id), canonical, *data.get("aliases", [])]
            normalized_candidates = [normalize_token(c) for c in candidates if c]

            score = 0.0
            for candidate in normalized_candidates:
                if not candidate:
                    continue
                # Exact or substring containment boost
                if wanted == candidate:
                    sim = 1.0
                elif wanted in candidate or candidate in wanted:
                    sim = 0.92
                else:
                    cand_tokens = token_set(candidate)
                    if wanted_tokens and cand_tokens and wanted_tokens.issubset(cand_tokens):
                        sim = 0.88
                    else:
                        sim = difflib.SequenceMatcher(None, wanted, candidate).ratio()
                if sim > score:
                    score = sim

            if score > best_score:
                best_score = score
                best_node = node_id

        if best_node is not None and best_score >= cutoff:
            canonical_name = graph.nodes[best_node].get("canonical_name", str(best_node))
            matches.append({
                "anchor": anchor,
                "node_id": best_node,
                "canonical_name": canonical_name,
                "score": round(best_score, 4),
            })
            seen_nodes.add(best_node)

    return matches


def relation_matches(relation: str, relation_anchors: list[str]) -> bool:
    """Check if an edge relation matches any of the query relation anchors."""
    if not relation_anchors:
        return True
    rel_tokens = token_set(relation.replace("_", " "))
    for anchor in relation_anchors:
        anchor_tokens = token_set(anchor)
        if rel_tokens & anchor_tokens:
            return True
        if any(a.startswith(r) or r.startswith(a) for a in anchor_tokens for r in rel_tokens):
            return True
    return False


@dataclass
class PathStep:
    source: str
    target: str
    edge_key: str
    relation: str
    claim_id: str | None
    inferred: bool
    confidence: float
    flow: float
    supporting_text: str = ""


@dataclass
class RelationalPath:
    nodes: list[str]
    steps: list[PathStep]
    reliability: float
    hop_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "hop_count": self.hop_count,
            "reliability": round(self.reliability, 4),
            "steps": [
                {
                    "source": step.source,
                    "target": step.target,
                    "edge_key": step.edge_key,
                    "relation": step.relation,
                    "claim_id": step.claim_id,
                    "inferred": step.inferred,
                    "confidence": round(step.confidence, 4),
                    "flow": round(step.flow, 4),
                    "supporting_text": step.supporting_text,
                }
                for step in self.steps
            ],
        }


class PathRAGTraverser:
    """Flow-based path traverser and claim retriever following PathRAG principles.

    Attributes:
        alpha: Decay rate of information propagation along edges (default 0.8 per paper).
        theta: Early-stopping threshold to prune weak paths (default 0.05 per paper).
        max_hops: Maximum path hops between query entity endpoints (default 3).
        top_k: Number of most reliable relational paths to retain (default 15).
        include_context_bridges: Whether navigation bridges may be traversed (bridges
                                 are still never returned as factual claims).
    """

    def __init__(
        self,
        alpha: float = 0.8,
        theta: float = 0.05,
        max_hops: int = 3,
        top_k: int = 15,
        include_context_bridges: bool = True,
    ) -> None:
        self.alpha = alpha
        self.theta = theta
        self.max_hops = max_hops
        self.top_k = top_k
        self.include_context_bridges = include_context_bridges

    def _edge_factor(
        self,
        data: dict[str, Any],
        relation_anchors: list[str] | None,
    ) -> float:
        """Compute edge propagation factor incorporating confidence and query relation bonus."""
        confidence = float(data.get("confidence", 1.0))
        inferred = bool(data.get("inferred", False))

        # Context bridges receive a penalty during path traversal
        if inferred or data.get("relation") in CONTEXT_RELATIONS:
            return 0.4 if self.include_context_bridges else 0.0

        factor = confidence
        if relation_anchors and relation_matches(data.get("relation", ""), relation_anchors):
            factor *= 1.25  # Relation alignment bonus
        return factor

    def propagate_flow(
        self,
        graph: nx.MultiDiGraph,
        start_node: str,
        relation_anchors: list[str] | None = None,
    ) -> dict[str, float]:
        """Compute resource flow from start_node with distance decay and early pruning.

        Flow formulation: S(v) = alpha * sum_{u in in_neighbors} S(u) / deg_out(u).
        Prunes branches where S(v) < theta.
        """
        resources: dict[str, float] = defaultdict(float)
        resources[start_node] = 1.0

        queue: deque[tuple[str, int]] = deque([(start_node, 0)])
        visited: set[str] = {start_node}

        while queue:
            curr_node, depth = queue.popleft()
            if depth >= self.max_hops:
                continue

            curr_res = resources[curr_node]
            # Out-degree over valid traversable edges
            neighbors = list(graph.successors(curr_node)) + list(graph.predecessors(curr_node))
            deg_out = max(1, len(set(neighbors)))

            # Check outbound and inbound edges (bidirectional reasoning over knowledge graph)
            out_candidates: list[tuple[str, str, str, dict[str, Any]]] = []
            for target in graph.successors(curr_node):
                for key, data in graph[curr_node][target].items():
                    out_candidates.append((curr_node, target, key, data))
            for source in graph.predecessors(curr_node):
                for key, data in graph[source][curr_node].items():
                    out_candidates.append((curr_node, source, key, data))

            for u, v, key, data in out_candidates:
                if v in visited:
                    continue

                edge_mult = self._edge_factor(data, relation_anchors)
                if edge_mult <= 0:
                    continue

                # Propagate flow with decay
                flow_delta = self.alpha * (curr_res / deg_out) * edge_mult

                # Early stopping: discard flow below pruning threshold theta
                if flow_delta < self.theta:
                    continue

                resources[v] = flow_delta
                visited.add(v)
                queue.append((v, depth + 1))

        return resources


    def find_paths_between(
        self,
        graph: nx.MultiDiGraph,
        source: str,
        target: str,
        flow_map: dict[str, float],
        relation_anchors: list[str] | None = None,
    ) -> list[RelationalPath]:
        """Find paths connecting source to target guided by flow and bounded by max_hops."""
        if source == target or source not in graph or target not in graph:
            return []

        # Undirected search space representation for traversal
        undirected_adj: dict[str, list[tuple[str, str, str, dict[str, Any]]]] = defaultdict(list)
        for u, v, key, data in graph.edges(keys=True, data=True):
            mult = self._edge_factor(data, relation_anchors)
            if mult > 0:
                undirected_adj[u].append((u, v, key, data))
                undirected_adj[v].append((v, u, key, data))

        # BFS path enumeration bounded by max_hops and flow threshold
        found_paths: list[RelationalPath] = []
        queue: deque[tuple[list[str], list[PathStep], float]] = deque([
            ([source], [], flow_map.get(source, 1.0))
        ])

        while queue:
            curr_nodes, curr_steps, curr_flow_acc = queue.popleft()
            curr_node = curr_nodes[-1]

            if curr_node == target and curr_steps:
                reliability = curr_flow_acc / len(curr_steps)
                found_paths.append(RelationalPath(
                    nodes=curr_nodes,
                    steps=curr_steps,
                    reliability=reliability,
                    hop_count=len(curr_steps),
                ))
                continue

            if len(curr_steps) >= self.max_hops:
                continue

            for from_node, next_node, key, data in undirected_adj.get(curr_node, []):
                if next_node in curr_nodes:
                    continue  # Simple path (no cycles)

                edge_factor = self._edge_factor(data, relation_anchors)
                edge_flow = flow_map.get(next_node, 0.0) * edge_factor

                # Skip if edge contributes negligible flow
                if edge_flow < (self.theta * 0.5):
                    continue

                step = PathStep(
                    source=from_node,
                    target=next_node,
                    edge_key=key,
                    relation=data.get("relation", ""),
                    claim_id=data.get("claim_id"),
                    inferred=bool(data.get("inferred", False)),
                    confidence=float(data.get("confidence", 1.0)),
                    flow=edge_flow,
                    supporting_text=data.get("supporting_text", ""),
                )

                queue.append((
                    curr_nodes + [next_node],
                    curr_steps + [step],
                    curr_flow_acc + edge_flow,
                ))

        return sorted(found_paths, key=lambda p: p.reliability, reverse=True)

    def retrieve_paths(
        self,
        graph: nx.MultiDiGraph,
        matched_node_ids: list[str],
        relation_anchors: list[str] | None = None,
    ) -> list[RelationalPath]:
        """Retrieve top-K relational paths across pairs of query-matched nodes.

        If only one node is matched, expands multi-hop paths to its highest-flow neighbors.
        """
        all_paths: list[RelationalPath] = []

        if len(matched_node_ids) >= 2:
            # Multi-hop paths between distinct query entity pairs
            for i, src in enumerate(matched_node_ids):
                flow_map = self.propagate_flow(graph, src, relation_anchors)
                for dst in matched_node_ids[i + 1:]:
                    pair_paths = self.find_paths_between(
                        graph, src, dst, flow_map, relation_anchors
                    )
                    all_paths.extend(pair_paths)

        # Fallback to single-seed / neighborhood expansion if no inter-pair paths formed
        if not all_paths and matched_node_ids:
            for src in matched_node_ids:
                flow_map = self.propagate_flow(graph, src, relation_anchors)
                high_flow_targets = [
                    node for node, flow in flow_map.items()
                    if node != src and flow >= self.theta
                ]
                for dst in high_flow_targets:
                    pair_paths = self.find_paths_between(
                        graph, src, dst, flow_map, relation_anchors
                    )
                    all_paths.extend(pair_paths)

        # Sort all candidate paths by PathRAG reliability score S(P)
        all_paths.sort(key=lambda p: p.reliability, reverse=True)
        return all_paths[: self.top_k]


    def extract_and_deduplicate_claims(
        self,
        paths: list[RelationalPath],
        claims_store: dict[str, dict[str, Any]],
        graph: nx.MultiDiGraph | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """Extract factual claims from traversed paths and drop all duplicates.

        Drops:
        - Duplicate claims (deduplicated by claim_id and canonical triple).
        - Non-factual context bridges (inferred=True or relation in CONTEXT_RELATIONS).

        Returns:
            (deduplicated_claims, num_duplicates_dropped)
        """
        unique_claims: dict[str, dict[str, Any]] = {}
        duplicates_dropped = 0

        for path_rank, path in enumerate(paths, start=1):
            for step in path.steps:
                # 1. Strictly drop non-factual context bridges
                if step.inferred or (step.relation in CONTEXT_RELATIONS):
                    continue

                claim_id = step.claim_id
                if not claim_id:
                    continue

                # 2. Retrieve claim metadata from claims store or edge attributes
                base_claim = claims_store.get(claim_id)
                if base_claim is None and graph is not None:
                    # Fallback to reconstructing from edge data if claims store lacks it
                    base_claim = {
                        "claim_id": claim_id,
                        "subject": graph.nodes[step.source].get("canonical_name", step.source),
                        "subject_id": step.source,
                        "relation": step.relation,
                        "object": graph.nodes[step.target].get("canonical_name", step.target),
                        "object_id": step.target,
                        "confidence": step.confidence,
                        "supporting_text": step.supporting_text,
                        "fact": f"{step.source} {step.relation} {step.target}",
                    }

                if base_claim is None:
                    continue

                # Semantic fact key: (subject, relation, object)
                subj = str(base_claim.get("subject_id") or base_claim.get("subject", "")).casefold().strip()
                rel = str(base_claim.get("relation", "")).casefold().strip()
                obj = str(base_claim.get("object_id") or base_claim.get("object", "")).casefold().strip()
                dedup_key = f"{subj}::{rel}::{obj}" if (subj and rel and obj) else claim_id

                # 3. Deduplication: update if already seen, else insert
                if dedup_key in unique_claims:
                    duplicates_dropped += 1
                    existing = unique_claims[dedup_key]
                    existing["path_count"] += 1
                    existing.setdefault("occurrences", []).append({
                        "claim_id": claim_id,
                        "document_id": base_claim.get("document_id"),
                        "sentence_span": base_claim.get("sentence_span"),
                        "supporting_text": base_claim.get("supporting_text"),
                    })
                    # Keep maximum path reliability across all paths containing this claim
                    if path.reliability > existing["path_reliability"]:
                        existing["path_reliability"] = round(path.reliability, 4)
                        existing["best_path_rank"] = path_rank
                    if float(base_claim.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                        existing["confidence"] = base_claim.get("confidence")
                        existing["supporting_text"] = base_claim.get("supporting_text")
                        existing["sentence_span"] = base_claim.get("sentence_span")
                        existing["document_id"] = base_claim.get("document_id")
                        existing["chunk_id"] = base_claim.get("chunk_id")
                else:
                    claim_record = dict(base_claim)
                    claim_record["path_reliability"] = round(path.reliability, 4)
                    claim_record["edge_flow"] = round(step.flow, 4)
                    claim_record["best_path_rank"] = path_rank
                    claim_record["path_count"] = 1
                    claim_record["occurrences"] = [{
                        "claim_id": claim_id,
                        "document_id": base_claim.get("document_id"),
                        "sentence_span": base_claim.get("sentence_span"),
                        "supporting_text": base_claim.get("supporting_text"),
                    }]
                    unique_claims[dedup_key] = claim_record

        # Sort deduplicated claims by path reliability and extraction confidence
        deduplicated = sorted(
            unique_claims.values(),
            key=lambda c: (c.get("path_reliability", 0.0), float(c.get("confidence", 0.0))),
            reverse=True,
        )

        return deduplicated, duplicates_dropped



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
                rel_bonus = 0.5 if (relation_anchors and relation_matches(c.get("relation", ""), relation_anchors)) else 0.0
                subj = str(c.get("subject_id") or c.get("subject", "")).casefold().strip()
                rel = str(c.get("relation", "")).casefold().strip()
                obj = str(c.get("object_id") or c.get("object", "")).casefold().strip()
                dedup_key = f"{subj}::{rel}::{obj}" if (subj and rel and obj) else cid

                if dedup_key in seen_facts:
                    existing = seen_facts[dedup_key]
                    existing["path_count"] += 1
                    existing.setdefault("occurrences", []).append({
                        "claim_id": cid,
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



    return claims


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
                rel_bonus = 0.5 if (relation_anchors and relation_matches(c.get("relation", ""), relation_anchors)) else 0.0
                subj = str(c.get("subject_id") or c.get("subject", "")).casefold().strip()
                rel = str(c.get("relation", "")).casefold().strip()
                obj = str(c.get("object_id") or c.get("object", "")).casefold().strip()
                dedup_key = f"{subj}::{rel}::{obj}" if (subj and rel and obj) else cid

                if dedup_key in seen_facts:
                    duplicates_dropped += 1
                    existing = seen_facts[dedup_key]
                    existing["path_count"] += 1
                    existing.setdefault("occurrences", []).append({
                        "claim_id": cid,
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




    return {
        "matched_nodes": matches,
        "paths": [p.to_dict() for p in paths],
        "claims": claims,
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
