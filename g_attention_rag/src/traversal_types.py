"""Shared matching, polarity, and path record types for retrieval."""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

import networkx as nx

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

def claim_dedup_key(claim: dict[str, Any], fallback_id: str) -> str:
    """Deduplicate repeated evidence without merging opposing assertions."""
    subject = str(claim.get("subject_id") or claim.get("subject", "")).casefold().strip()
    relation = str(claim.get("relation", "")).casefold().strip()
    obj = str(claim.get("object_id") or claim.get("object", "")).casefold().strip()
    if not (subject and relation and obj):
        return fallback_id
    polarity = "negated" if bool(claim.get("negated", False)) else "affirmed"
    return f"{subject}::{relation}::{obj}::{polarity}"

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
    negated: bool = False

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
                    "negated": step.negated,
                    "inferred": step.inferred,
                    "confidence": round(step.confidence, 4),
                    "flow": round(step.flow, 4),
                    "supporting_text": step.supporting_text,
                }
                for step in self.steps
            ],
        }
