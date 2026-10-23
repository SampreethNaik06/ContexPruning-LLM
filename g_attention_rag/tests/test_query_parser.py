"""Regression checks for query anchors and persistent query logs."""

import json
from pathlib import Path
import tempfile
import unittest

from g_attention_rag.src.query_parser import StandaloneQueryParser


class QueryParserTest(unittest.TestCase):
    def test_anchors_and_log_append(self):
        parser = StandaloneQueryParser()
        result = parser.parse_query("Who explained the photoelectric effect in 1905?")
        self.assertEqual(set(result["anchors"]["entity_anchors"]), {"1905"})
        self.assertEqual(set(result["anchors"]["relation_anchors"]), {"explain", "effect"})
        self.assertEqual(result["detailed_log"]["anchors"], result["anchors"])
        self.assertEqual(
            parser.parse_query("")["anchors"],
            {"entity_anchors": [], "relation_anchors": []},
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.json"
            for expected_count in (1, 2):
                parser.save_log_to_disk(result["detailed_log"], str(path))
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["metadata"]["total_queries_logged"], expected_count)
                self.assertEqual(payload["logs"], [result["detailed_log"]] * expected_count)

            path.write_text("invalid JSON", encoding="utf-8")
            parser.save_log_to_disk(result["detailed_log"], str(path))
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["logs"], [result["detailed_log"]])


if __name__ == "__main__":
    unittest.main()
