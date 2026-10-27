"""Adapt the existing query parser and PathRAG retrieval for chat."""

import pickle
import re
from threading import Lock

from main import _graph_entity_mentions
from src.graph_constants import SCHEMA_VERSION
from src.graph_traversal import retrieve_with_traversal_details
from src.query_parser import StandaloneQueryParser

from .settings import GRAPH_PATH


class Retriever:
    """Lazily load the local graph and parser once for this server."""

    def __init__(self) -> None:
        self.state = None
        self.parser = None
        # ponytail: serialize local retrieval; use worker processes for multi-user hosting.
        self.lock = Lock()

    def retrieve(self, query: str, history: list[dict]) -> dict:
        """Return fresh claims and supporting passages for each question."""
        with self.lock:
            if self.state is None:
                if not GRAPH_PATH.exists():
                    raise RuntimeError("No knowledge graph found. Build data/knowledge_graph.pkl first.")
                # Only load the project's own graph, never a path supplied by a client.
                with GRAPH_PATH.open("rb") as handle:
                    state = pickle.load(handle)
                if state.get("schema_version") != SCHEMA_VERSION:
                    raise RuntimeError("The knowledge graph is outdated. Rebuild it before chatting.")
                try:
                    parser = StandaloneQueryParser()
                except OSError as exc:
                    raise RuntimeError("Install the language model: python -m spacy download en_core_web_sm") from exc
                self.state, self.parser = state, parser

            anchors = self.parser.parse_query(query)["anchors"]
            mentions = _graph_entity_mentions(self.state, query)
            # Short follow-ups inherit the previous question's entities, not its answer.
            if history and not anchors["entity_anchors"] and not mentions:
                previous = next((m["text"] for m in reversed(history) if m["role"] == "user"), "")
                previous_anchors = self.parser.parse_query(previous)["anchors"]
                anchors["entity_anchors"] = previous_anchors["entity_anchors"]
                mentions = _graph_entity_mentions(self.state, previous)
            entities = list(dict.fromkeys([*anchors["entity_anchors"], *mentions]))
            entity_words = {
                word.rstrip("s") for mention in mentions
                for word in re.findall(r"[\w'-]+", mention.casefold())
            }
            relations = [
                word for word in anchors["relation_anchors"]
                if word.casefold().rstrip("s") not in entity_words
            ]
            return retrieve_with_traversal_details(
                self.state, entity_anchors=entities, relation_anchors=relations,
            )
