"""PathRAG flow propagation and bounded relational-path enumeration."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import networkx as nx

from .traversal_types import (
    CONTEXT_RELATIONS,
    PathStep,
    RelationalPath,
    claim_dedup_key,
    relation_matches,
)


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
                    negated=bool(data.get("negated", False)),
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
        - Duplicate claims (deduplicated by canonical triple and polarity).
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
                        "negated": step.negated,
                        "supporting_text": step.supporting_text,
                        "fact": f"{step.source} {step.relation} {step.target}",
                    }

                if base_claim is None:
                    continue

                # Keep affirmative and negated versions of a triple distinct.
                claim_for_key = dict(base_claim)
                claim_for_key.setdefault("negated", step.negated)
                dedup_key = claim_dedup_key(claim_for_key, claim_id)

                # 3. Deduplication: update if already seen, else insert
                if dedup_key in unique_claims:
                    duplicates_dropped += 1
                    existing = unique_claims[dedup_key]
                    existing["path_count"] += 1
                    existing.setdefault("occurrences", []).append({
                        "claim_id": claim_id,
                        "negated": bool(base_claim.get("negated", step.negated)),
                        "document_id": base_claim.get("document_id"),
                        "sentence_span": base_claim.get("sentence_span"),
                        "supporting_text": base_claim.get("supporting_text"),
                    })
                    # Keep maximum path reliability across all paths containing this claim
                    if path.reliability > existing["path_reliability"]:
                        existing["path_reliability"] = round(path.reliability, 4)
                        existing["best_path_rank"] = path_rank
                    claim_confidence = float(base_claim.get("confidence", 0.0))
                    existing_confidence = float(existing.get("confidence", 0.0))
                    if claim_confidence > existing_confidence:
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
                        "negated": bool(base_claim.get("negated", step.negated)),
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
