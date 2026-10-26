
"""Run from the repository root: python -B -m unittest discover -s g_attention_rag/tests."""

import json
from pathlib import Path
import tempfile
import unittest

import networkx as nx

from g_attention_rag.src.graph_builder import KnowledgeGraphBuilder, export_json


class GraphBuilderIntegrationTest(unittest.TestCase):
    def test_opposing_claims_in_one_sentence_remain_separate(self):
        builder = KnowledgeGraphBuilder()
        builder.add_chunk(
            "opposing-claims",
            "Microsoft acquired Nuance and did not acquire Nuance.",
        )

        claims = list(builder.state["claims"].values())
        nuance_claims = [
            claim for claim in claims
            if claim["subject_id"] == "microsoft" and claim["object_id"] == "nuance"
        ]

        self.assertEqual({claim["negated"] for claim in nuance_claims}, {False, True})
        self.assertEqual(len({claim["claim_id"] for claim in nuance_claims}), 2)
        self.assertTrue(all(
            claim["supporting_text"] == "Microsoft acquired Nuance and did not acquire Nuance."
            for claim in nuance_claims
        ))

    def test_discontinuous_phrase_mentions_keep_exact_spans_and_sentence_fallback(self):
        text = "The company developed state of the art technology."
        builder = KnowledgeGraphBuilder()
        builder.add_chunk("discontinuous-phrase", text)

        claim = next(
            claim for claim in builder.state["claims"].values()
            if claim["object"] == "state of art technology"
        )
        expected_spans = [[22, 30], [35, 49]]
        self.assertEqual(claim["object_spans"], expected_spans)
        self.assertIsNone(claim["object_span"])
        self.assertEqual(
            " ".join(text[start:end] for start, end in claim["object_spans"]),
            claim["object"],
        )
        mention = next(
            mention for mention in builder.graph.nodes[claim["object_id"]]["mentions"]
            if mention["text"] == claim["object"]
        )
        self.assertEqual(mention["spans"], expected_spans)
        self.assertIsNone(mention["span"])
        self.assertEqual(mention["sentence_span"], [0, len(text)])
        self.assertEqual(mention["supporting_text"], text)

    def test_claims_survive_document_merge_and_serialization(self):
        builder = KnowledgeGraphBuilder()
        builder.add_chunk("active", "Microsoft acquired Nuance. Microsoft owns Nuance.")
        builder.add_chunk("passive", "Nuance was acquired by Microsoft. Google acquired YouTube.")
        state = builder.state
        graph = builder.graph
        self.assertIsInstance(graph, nx.MultiDiGraph)
        self.assertEqual(len(state["documents"]), 2)
        self.assertGreaterEqual(graph.number_of_edges("microsoft", "nuance"), 3)
        self.assertEqual(state["traversal_graph"].number_of_edges(), len(graph) - 1)
        for claim_id, claim in state["claims"].items():
            document = state["documents"][claim["document_id"]]["text"]
            start, end = claim["sentence_span"]
            self.assertEqual(document[start:end], claim["supporting_text"])
            chunk = state["chunks"][claim["chunk_id"]]
            self.assertEqual(document[chunk["start_char"]:chunk["end_char"]], chunk["text"])
            for node in (claim["subject_id"], claim["object_id"]):
                self.assertIn(claim_id, graph.nodes[node]["claim_ids"])
        with tempfile.TemporaryDirectory() as temporary:
            pickle_path = Path(temporary) / "graph.pkl"
            json_path = Path(temporary) / "graph.json"
            builder.save(pickle_path)
            restored = KnowledgeGraphBuilder.load(pickle_path)
            export_json(restored.state, json_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(set(payload["claims"]), set(state["claims"]))
            self.assertEqual(restored.stats(), builder.stats())

    def test_alias_from_one_document_canonicalizes_other_documents(self):
        sources = [
            ("full-name", "International Business Machines (IBM) acquired Red Hat."),
            ("acronym", "IBM acquired HashiCorp."),
        ]
        for ordered_sources in (sources, list(reversed(sources))):
            with self.subTest(first_source=ordered_sources[0][0]):
                builder = KnowledgeGraphBuilder()
                for source_name, text in ordered_sources:
                    builder.add_chunk(source_name, text)

                canonical_id = builder.state["explicit_aliases"]["ibm"]
                self.assertEqual(canonical_id, "international business machine")
                self.assertIn(canonical_id, builder.graph)
                self.assertNotIn("ibm", builder.graph)
                self.assertEqual(
                    builder.graph.nodes[canonical_id]["canonical_name"],
                    "International Business Machines",
                )
                self.assertIn("IBM", builder.graph.nodes[canonical_id]["aliases"])
                acronym_document_id = next(
                    document_id for document_id, document in builder.state["documents"].items()
                    if document["source_name"] == "acronym"
                )
                acronym_claims = [
                    claim for claim in builder.state["claims"].values()
                    if claim["document_id"] == acronym_document_id
                ]
                self.assertTrue(acronym_claims)
                self.assertTrue(all(
                    claim["subject_id"] == canonical_id for claim in acronym_claims
                ))


if __name__ == "__main__":
    unittest.main()
