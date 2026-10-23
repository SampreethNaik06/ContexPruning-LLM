
"""Run from the repository root: python -B -m unittest discover -s g_attention_rag/tests."""

import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from g_attention_rag.eval.evaluate import evaluate, retrieve
from g_attention_rag.src.graph_builder import KnowledgeGraphBuilder, export_json


class GraphBuilderIntegrationTest(unittest.TestCase):
    def test_claims_survive_document_merge_and_serialization(self):
        builder = KnowledgeGraphBuilder()
        builder.add_chunk("active", "Microsoft acquired Nuance. Microsoft owns Nuance.")
        builder.add_chunk("passive", "Nuance was acquired by Microsoft. Google acquired YouTube.")
        state = builder.state
        graph = builder.graph
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertEqual(len(state["documents"]), 2)
        self.assertGreaterEqual(graph.number_of_edges("microsoft", "nuance"), 3)
        self.assertTrue(evaluate(state)["all_passed"])
        self.assertEqual(state["traversal_graph"].number_of_edges(), len(graph) - 1)
        for claim_id, claim in state["claims"].items():
            document = state["documents"][claim["document_id"]]["text"]
            start, end = claim["sentence_span"]
            self.assertEqual(document[start:end], claim["supporting_text"])
            chunk = state["chunks"][claim["chunk_id"]]
            self.assertEqual(document[chunk["start_char"]:chunk["end_char"]], chunk["text"])
            for node in (claim["subject_id"], claim["object_id"]):
                self.assertIn(claim_id, graph.nodes[node]["claim_ids"])
        result = retrieve(state, ["Microsoft"], ["acquire"])
        self.assertTrue(result["claims"])
        self.assertTrue(all(claim["claim_id"] in state["claims"] for claim in result["claims"]))
        with tempfile.TemporaryDirectory() as temporary:
            pickle_path = Path(temporary) / "graph.pkl"
            json_path = Path(temporary) / "graph.json"
            builder.save(pickle_path)
            restored = KnowledgeGraphBuilder.load(pickle_path)
            export_json(restored.state, json_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["claims"]), set(state["claims"]))
            self.assertEqual(restored.stats(), builder.stats())
            self.assertTrue(evaluate(restored.state)["all_passed"])


if __name__ == "__main__":
    unittest.main()
