"""Quality and retrieval checks for an ECAP entity-anchored graph."""

from __future__ import annotations

import argparse
import difflib
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Any

import networkx as nx


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def simple_stem(token: str) -> str:
    """Small dependency-free normalizer for query anchors and predicates."""
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith(("ated", "ized")) and len(token) > 5:
        return token[:-1]
    if token.endswith("ing") and len(token) > 5:
        return token[:-3]
    if token.endswith("ed") and len(token) > 4:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def normalized_tokens(value: str) -> set[str]:
    return {simple_stem(token) for token in normalize(value).split()}


def load_state(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def anchors_from_reference(path: Path, log_index: int = -1) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "logs" in payload:
        anchors = payload["logs"][log_index]["anchors"]
    else:
        anchors = payload.get("anchors", payload)
    entities = anchors.get("entity_anchors", anchors.get("entities", []))
    relations = anchors.get("relation_anchors", anchors.get("relations", []))
    return list(entities), list(relations)


def match_nodes(graph: nx.MultiDiGraph, anchors: list[str], cutoff: float = 0.68):
    results = []
    for anchor in anchors:
        wanted = normalize(anchor)
        best_node, best_score = None, 0.0
        for node, data in graph.nodes(data=True):
            candidates = [str(node), data.get("canonical_name", ""), *data.get("aliases", [])]
            normalized_candidates = [normalize(candidate) for candidate in candidates if candidate]
            score = max(
                difflib.SequenceMatcher(None, wanted, candidate).ratio()
                for candidate in normalized_candidates
            )
            if wanted and any(wanted in candidate or candidate in wanted for candidate in normalized_candidates):
                score = max(score, 0.9)
            if score > best_score:
                best_node, best_score = node, score
        if best_node is not None and best_score >= cutoff:
            results.append({"anchor": anchor, "node_id": best_node, "score": round(best_score, 3)})
    return results


def relation_matches(relation: str, anchors: list[str]) -> bool:
    if not anchors:
        return True
    relation_tokens = normalized_tokens(relation)
    for anchor in anchors:
        anchor_tokens = normalized_tokens(anchor)
        if relation_tokens & anchor_tokens:
            return True
        if any(a.startswith(r) or r.startswith(a) for a in anchor_tokens for r in relation_tokens):
            return True
    return False


def claim_matches_anchors(claim: dict[str, Any], anchors: list[str]) -> bool:
    if relation_matches(claim.get("relation", ""), anchors):
        return True
    searchable = normalized_tokens(
        " ".join([
            claim.get("subject", ""), claim.get("object", ""),
            claim.get("supporting_text", ""),
        ])
    )
    return any(normalized_tokens(anchor) & searchable for anchor in anchors if normalize(anchor))


def reasoning_paths(
    graph: nx.MultiDiGraph,
    matched_node_ids: list[str],
    relation_anchors: list[str],
    max_hops: int,
    limit: int = 25,
) -> list[dict[str, Any]]:
    """Return bounded paths between query entities with provenance-bearing steps."""
    if len(matched_node_ids) < 2:
        return []
    simple = nx.Graph()
    simple.add_nodes_from(graph.nodes)
    simple.add_edges_from((source, target) for source, target in graph.edges())
    results: list[dict[str, Any]] = []
    for left_index, source in enumerate(matched_node_ids):
        for target in matched_node_ids[left_index + 1:]:
            if source == target or not nx.has_path(simple, source, target):
                continue
            for nodes in nx.all_simple_paths(simple, source, target, cutoff=max_hops):
                steps = []
                path_score = 0.0
                for left, right in zip(nodes, nodes[1:]):
                    candidates = []
                    for edge_source, edge_target in ((left, right), (right, left)):
                        if not graph.has_edge(edge_source, edge_target):
                            continue
                        for key, data in graph[edge_source][edge_target].items():
                            relation_bonus = 0.25 if relation_matches(
                                data.get("relation", ""), relation_anchors
                            ) else 0.0
                            context_penalty = 0.35 if data.get("inferred") else 0.0
                            score = float(data.get("confidence", 0.0)) + relation_bonus - context_penalty
                            candidates.append((score, edge_source, edge_target, key, data))
                    if not candidates:
                        steps = []
                        break
                    score, edge_source, edge_target, key, data = max(candidates, key=lambda item: item[0])
                    path_score += score
                    steps.append({
                        "source": edge_source,
                        "target": edge_target,
                        "edge_key": key,
                        "relation": data.get("relation"),
                        "claim_id": data.get("claim_id"),
                        "inferred": bool(data.get("inferred")),
                        "confidence": data.get("confidence"),
                        "supporting_text": data.get("supporting_text", ""),
                    })
                if steps:
                    results.append({
                        "nodes": nodes,
                        "steps": steps,
                        "score": round(path_score / len(steps), 4),
                    })
                if len(results) >= limit:
                    return sorted(results, key=lambda item: item["score"], reverse=True)
    return sorted(results, key=lambda item: item["score"], reverse=True)


def retrieve(state: dict[str, Any], entities: list[str], relations: list[str],
             neighbor_depth: int = 1, max_path_hops: int = 3) -> dict[str, Any]:
    graph: nx.MultiDiGraph = state["graph"]
    traversal_graph: nx.MultiDiGraph = state.get("traversal_graph", graph)
    matches = match_nodes(graph, entities)
    matched_nodes = {item["node_id"] for item in matches}
    selected_nodes = set(matched_nodes)
    frontier = set(selected_nodes)
    undirected = traversal_graph.to_undirected()
    for _ in range(neighbor_depth):
        frontier = {neighbor for node in frontier for neighbor in undirected.neighbors(node)} - selected_nodes
        selected_nodes.update(frontier)

    found_claims: dict[str, dict[str, Any]] = {}
    for node in selected_nodes:
        for claim in graph.nodes[node].get("claims", []):
            if claim_matches_anchors(claim, relations):
                found_claims[claim["claim_id"]] = claim

    ranked = sorted(
        found_claims.values(),
        key=lambda claim: (
            claim["subject_id"] in matched_nodes or claim["object_id"] in matched_nodes,
            float(claim.get("confidence", 0.0)),
        ),
        reverse=True,
    )
    return {
        "entity_anchors": entities,
        "relation_anchors": relations,
        "matched_nodes": matches,
        "claims": ranked,
        "reasoning_paths": reasoning_paths(
            traversal_graph,
            [match["node_id"] for match in matches], relations, max_path_hops
        ),
        "traversal_view": "sparse_backbone" if "traversal_graph" in state else "full_graph",
    }


FORBIDDEN_NODES = {
    "this", "that", "which", "one", "other", "another", "these questions",
    "this principle", "this paradox", "this discovery",
}

RETRIEVAL_CASES = [
    {
        "name": "Heisenberg formulated uncertainty principle",
        "entities": ["uncertainty principle"],
        "relations": ["formulate"],
        "evidence_terms": ["Werner Heisenberg"],
    },
    {
        "name": "Einstein explained photoelectric effect",
        "entities": ["Albert Einstein"],
        "relations": ["explain"],
        "evidence_terms": ["photoelectric effect", "1905"],
    },
    {
        "name": "Quantum mechanics underpins technologies",
        "entities": ["quantum mechanics"],
        "relations": ["underpin"],
        "evidence_terms": ["semiconductors", "lasers"],
    },
    {
        "name": "Quantum computing leverages superposition",
        "entities": ["quantum computing"],
        "relations": ["leverage"],
        "evidence_terms": ["superposition", "entanglement"],
    },
]


def evaluate(state: dict[str, Any], source_text: str | None = None, *, fixture_checks: bool = False) -> dict[str, Any]:
    graph: nx.MultiDiGraph = state["graph"]
    traversal_graph: nx.MultiDiGraph = state.get("traversal_graph", graph)
    results: list[dict[str, Any]] = []

    component_count = nx.number_weakly_connected_components(graph)
    results.append({
        "name": "single connected component",
        "passed": component_count == 1,
        "detail": f"component_count={component_count}",
    })

    traversal_components = nx.number_weakly_connected_components(traversal_graph)
    expected_edges = max(0, traversal_graph.number_of_nodes() - 1)
    results.append({
        "name": "sparse connected traversal backbone",
        "passed": (
            traversal_components == 1
            and traversal_graph.number_of_edges() == expected_edges
        ),
        "detail": (
            f"nodes={traversal_graph.number_of_nodes()}, "
            f"edges={traversal_graph.number_of_edges()}, "
            f"components={traversal_components}"
        ),
    })

    present = {
        normalize(data.get("canonical_name", str(node)))
        for node, data in graph.nodes(data=True)
    }
    bad = sorted(present & {normalize(item) for item in FORBIDDEN_NODES})
    results.append({
        "name": "no unresolved generic/deictic nodes",
        "passed": not bad,
        "detail": f"forbidden_nodes={bad}",
    })

    if fixture_checks:
        alias_nodes = [
            data for _, data in graph.nodes(data=True)
            if {normalize(alias) for alias in data.get("aliases", [])}
            >= {"quantum physics", "quantum mechanics"}
        ]
        results.append({
            "name": "quantum physics/mechanics alias merge",
            "passed": len(alias_nodes) == 1,
            "detail": f"matching_canonical_nodes={len(alias_nodes)}",
        })

    provenance_errors: list[str] = []
    for claim_id, claim in state.get("claims", {}).items():
        start, end = claim["sentence_span"]
        document = state.get("documents", {}).get(claim["document_id"], {})
        evidence_source = document.get("text", source_text or state.get("source_text", ""))
        if evidence_source[start:end].strip() != claim["supporting_text"].strip():
            provenance_errors.append(claim_id)
        if claim["subject_id"] not in graph or claim["object_id"] not in graph:
            provenance_errors.append(claim_id)
    results.append({
        "name": "claim provenance round trip",
        "passed": not provenance_errors,
        "detail": f"invalid_claims={sorted(set(provenance_errors))[:10]}",
    })

    for case in RETRIEVAL_CASES if fixture_checks else []:
        retrieval = retrieve(
            state, case["entities"], case["relations"], neighbor_depth=1
        )
        evidence = " ".join(
            claim.get("supporting_text", "") for claim in retrieval["claims"]
        ).casefold()
        missing = [term for term in case["evidence_terms"] if term.casefold() not in evidence]
        results.append({
            "name": case["name"],
            "passed": bool(retrieval["matched_nodes"] and retrieval["claims"] and not missing),
            "detail": (
                f"matched_nodes={len(retrieval['matched_nodes'])}, "
                f"claims={len(retrieval['claims'])}, missing_terms={missing}"
            ),
        })

    passed = sum(result["passed"] for result in results)
    return {
        "passed": passed,
        "total": len(results),
        "all_passed": passed == len(results),
        "results": results,
    }


def notebook_safe_argv(argv: list[str] | None = None) -> list[str]:
    values = list(sys.argv[1:] if argv is None else argv)
    cleaned: list[str] = []
    index = 0
    while index < len(values):
        if values[index] in {"-f", "--f"}:
            index += 2
            continue
        cleaned.append(values[index])
        index += 1
    return cleaned


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", nargs="?", type=Path, default=Path("data/knowledge_graph.pkl"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--fixture-checks", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when a check fails")
    args = parser.parse_args(notebook_safe_argv(argv))

    report = evaluate(
        load_state(args.graph),
        args.source.read_text(encoding="utf-8").strip() if args.source else None,
        fixture_checks=args.fixture_checks
    )
    for result in report["results"]:
        marker = "PASS" if result["passed"] else "FAIL"
        print(f"[{marker}] {result['name']}: {result['detail']}")
    print(f"\nPassed {report['passed']}/{report['total']} checks")
    if args.strict and not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
