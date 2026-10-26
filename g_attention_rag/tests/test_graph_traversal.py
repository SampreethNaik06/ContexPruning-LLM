"""Unit and integration tests for PathRAG-inspired graph traversal and claim retrieval.

Run from repository root:
python -B -m unittest discover -s g_attention_rag/tests
"""

import unittest
import networkx as nx

from g_attention_rag.src.graph_builder import KnowledgeGraphBuilder
from g_attention_rag.src.graph_traversal import (
    PathRAGTraverser,
    match_nodes,
    retrieve_claims,
    retrieve_with_traversal_details,
)


class TestPathRAGTraversal(unittest.TestCase):
    def setUp(self):
        # Build a multi-document graph with known multi-hop connections and multi-path overlaps
        self.builder = KnowledgeGraphBuilder()
        self.builder.add_chunk(
            "doc1",
            "Albert Einstein published theory of relativity in 1905. "
            "Theory of relativity revolutionized modern physics."
        )
        self.builder.add_chunk(
            "doc2",
            "Modern physics influenced quantum mechanics. "
            "Albert Einstein debated Niels Bohr about quantum mechanics."
        )
        self.state = self.builder.state
        self.graph = self.builder.graph

    def test_node_anchor_matching(self):
        """Test fuzzy and alias matching of entity anchors."""
        matches = match_nodes(self.graph, ["Einstein", "Relativity", "Quantum Mechanics"])
        matched_ids = [m["node_id"] for m in matches]
        self.assertTrue(any("einstein" in nid for nid in matched_ids))
        self.assertTrue(any("relativity" in nid for nid in matched_ids))
        self.assertTrue(any("quantum" in nid for nid in matched_ids))

    def test_flow_decay_and_pruning(self):
        """Test that resource flow diminishes with alpha and prunes when below theta."""
        traverser = PathRAGTraverser(alpha=0.8, theta=0.05, max_hops=3)
        # Find an existing node
        start_node = next(iter(self.graph.nodes))
        flow = traverser.propagate_flow(self.graph, start_node)
        self.assertIn(start_node, flow)
        self.assertEqual(flow[start_node], 1.0)
        # All positive flow values must be >= theta
        for node, val in flow.items():
            self.assertGreaterEqual(val, 0.0)

    def test_multi_hop_claims_retrieval(self):
        """Test finding multi-hop paths and extracting claims only."""
        claims = retrieve_claims(
            self.state,
            entity_anchors=["Albert Einstein", "Modern Physics"],
            alpha=0.8,
            theta=0.05,
            max_hops=3,
        )
        self.assertIsInstance(claims, list)
        self.assertGreater(len(claims), 0)

        # Ensure all retrieved items are factual claims
        for claim in claims:
            self.assertIn("claim_id", claim)
            self.assertIn("subject", claim)
            self.assertIn("relation", claim)
            self.assertIn("object", claim)
            self.assertIn("supporting_text", claim)
            self.assertIn("path_reliability", claim)
            # Ensure no context bridges were returned as claims
            self.assertNotIn(claim["relation"], {"covers", "contextualizes", "next_topic"})
            self.assertFalse(claim.get("inferred", False))

    def test_drop_duplicate_claims(self):
        """Test that duplicate claims across overlapping paths are strictly dropped."""
        details = retrieve_with_traversal_details(
            self.state,
            entity_anchors=["Albert Einstein", "Quantum Mechanics"],
            max_hops=3,
            top_k=20,
        )
        claims = details["claims"]
        stats = details["stats"]

        # Check unique claim IDs
        claim_ids = [c["claim_id"] for c in claims]
        self.assertEqual(
            len(claim_ids),
            len(set(claim_ids)),
            "Duplicate claim_ids found in retrieved claims list!",
        )

        # Check unique canonical facts (subject_id, relation, object_id)
        facts = [
            (c["subject_id"], c["relation"], c["object_id"], c.get("negated", False))
            for c in claims
        ]
        self.assertEqual(
            len(facts),
            len(set(facts)),
            "Duplicate facts found in retrieved claims list!",
        )

        # Check stats reporting
        self.assertIn("duplicates_dropped", stats)
        self.assertGreaterEqual(stats["duplicates_dropped"], 0)
        self.assertEqual(stats["claims_extracted"], len(claims))

    def test_retrieval_returns_original_source_chunks(self):
        """Retrieved claims link back to exact saved document chunks."""
        details = retrieve_with_traversal_details(
            self.state,
            entity_anchors=["Albert Einstein", "Modern Physics"],
            max_hops=3,
        )

        self.assertTrue(details["sources"])
        chunks_by_id = self.state["chunks"]
        documents = self.state["documents"]
        retrieved_sources = {source["chunk_id"]: source for source in details["sources"]}
        for claim in details["claims"]:
            self.assertIn("evidence", claim)
            for evidence in claim["evidence"]:
                chunk = chunks_by_id[evidence["chunk_id"]]
                document = documents[evidence["document_id"]]
                self.assertEqual(evidence["source_text"], chunk["text"])
                self.assertEqual(evidence["source_name"], document["source_name"])
                self.assertEqual(
                    retrieved_sources[evidence["chunk_id"]]["text"], chunk["text"]
                )

    def test_opposing_claims_survive_retrieval_deduplication(self):
        """Polarity is part of claim identity during path-level deduplication."""
        builder = KnowledgeGraphBuilder()
        builder.add_chunk(
            "opposing-retrieval",
            "Microsoft acquired Nuance. Microsoft did not acquire Nuance.",
        )

        details = retrieve_with_traversal_details(
            builder.state,
            entity_anchors=["Microsoft", "Nuance"],
            relation_anchors=["acquire"],
        )
        claims = [
            claim for claim in details["claims"]
            if claim["subject_id"] == "microsoft" and claim["object_id"] == "nuance"
        ]

        self.assertEqual({claim["negated"] for claim in claims}, {False, True})
        self.assertEqual(len({claim["claim_id"] for claim in claims}), 2)
        self.assertEqual(
            {evidence["negated"] for claim in claims for evidence in claim["evidence"]},
            {False, True},
        )

        graph_only_claims = retrieve_with_traversal_details(
            builder.graph,
            entity_anchors=["Microsoft", "Nuance"],
            relation_anchors=["acquire"],
        )["claims"]
        graph_only_conflicts = [
            claim for claim in graph_only_claims
            if claim["subject_id"] == "microsoft" and claim["object_id"] == "nuance"
        ]
        self.assertEqual(
            {claim["negated"] for claim in graph_only_conflicts},
            {False, True},
        )

    def test_opposing_claims_survive_direct_retrieval_fallback(self):
        """The no-path fallback uses the same polarity-aware claim identity."""
        graph = nx.MultiDiGraph()
        graph.add_node("microsoft", canonical_name="Microsoft", aliases=[], claims=[
            {
                "claim_id": "affirmed", "subject_id": "microsoft", "subject": "Microsoft",
                "relation": "acquire", "object_id": "nuance", "object": "Nuance",
                "negated": False, "confidence": 0.9, "supporting_text": "Microsoft acquired Nuance.",
            },
            {
                "claim_id": "negated", "subject_id": "microsoft", "subject": "Microsoft",
                "relation": "acquire", "object_id": "nuance", "object": "Nuance",
                "negated": True, "confidence": 0.9, "supporting_text": "Microsoft did not acquire Nuance.",
            },
        ])

        details = retrieve_with_traversal_details(
            {"graph": graph, "claims": {}},
            entity_anchors=["Microsoft"],
            relation_anchors=["acquire"],
        )

        self.assertEqual(
            {claim["negated"] for claim in details["claims"]},
            {False, True},
        )

    def test_synthetic_overlapping_paths_deduplication(self):
        """Construct synthetic multi-path graph where multiple routes share an edge."""
        g = nx.MultiDiGraph()
        # Paths:
        # A -> B -> D (claim 1 on A->B, claim 2 on B->D)
        # A -> C -> B -> D (claim 3 on A->C, claim 4 on C->B, claim 2 on B->D)
        g.add_node("a", canonical_name="Node A", aliases=[])
        g.add_node("b", canonical_name="Node B", aliases=[])
        g.add_node("c", canonical_name="Node C", aliases=[])
        g.add_node("d", canonical_name="Node D", aliases=[])

        claims_store = {
            "c_ab": {"claim_id": "c_ab", "subject_id": "a", "relation": "connects", "object_id": "b", "confidence": 0.95},
            "c_bd": {"claim_id": "c_bd", "subject_id": "b", "relation": "triggers", "object_id": "d", "confidence": 0.90},
            "c_ac": {"claim_id": "c_ac", "subject_id": "a", "relation": "links", "object_id": "c", "confidence": 0.85},
            "c_cb": {"claim_id": "c_cb", "subject_id": "c", "relation": "leads_to", "object_id": "b", "confidence": 0.88},
        }

        g.add_edge("a", "b", key="c_ab", claim_id="c_ab", relation="connects", confidence=0.95, inferred=False)
        g.add_edge("b", "d", key="c_bd", claim_id="c_bd", relation="triggers", confidence=0.90, inferred=False)
        g.add_edge("a", "c", key="c_ac", claim_id="c_ac", relation="links", confidence=0.85, inferred=False)
        g.add_edge("c", "b", key="c_cb", claim_id="c_cb", relation="leads_to", confidence=0.88, inferred=False)

        # Context bridge that MUST be ignored
        g.add_edge("a", "d", key="bridge_1", relation="covers", inferred=True, edge_type="document_context", confidence=0.5)

        state = {"graph": g, "claims": claims_store}
        details = retrieve_with_traversal_details(
            state,
            entity_anchors=["Node A", "Node D"],
            max_hops=3,
        )

        claims = details["claims"]
        stats = details["stats"]

        # Verify bridge_1 was excluded
        for c in claims:
            self.assertNotEqual(c.get("claim_id"), "bridge_1")
            self.assertNotEqual(c.get("relation"), "covers")

        # Verify claim c_bd was only included ONCE despite being traversed by both A->B->D and A->C->B->D
        bd_occurrences = [c for c in claims if c["claim_id"] == "c_bd"]
        self.assertEqual(len(bd_occurrences), 1)
        self.assertGreaterEqual(bd_occurrences[0]["path_count"], 2)
        self.assertGreater(stats["duplicates_dropped"], 0)


if __name__ == "__main__":
    unittest.main()
