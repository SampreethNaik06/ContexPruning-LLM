"""Manage corpus-wide aliases, document merges, and graph persistence."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import networkx as nx

from .document_graph import build_graph
from .entity_aliases import discover_corpus_aliases
from .graph_constants import SCHEMA_VERSION
from .graph_serialization import export_json, graph_to_dict
from .graph_structure import build_traversal_backbone, ensure_connected, graph_diagnostics
from .graph_text import clean_source_text, load_nlp, mask_heading_text, stable_id


class KnowledgeGraphBuilder:
    """Project-facing builder retaining the full Module 2 serialization contract."""

    def __init__(self, spacy_model: str = "en_core_web_sm", max_chunk_chars: int = 1200):
        self.spacy_model = spacy_model
        self.max_chunk_chars = max_chunk_chars
        self.nlp = None
        self.state = None
        self.graph = nx.MultiDiGraph()
        self._source_documents: dict[str, tuple[str, str]] = {}
        self._dirty = False

    def add_chunk(self, chunk_id: str, text: str, *, rebuild: bool = True) -> None:
        """Add a source, optionally deferring corpus-wide rebuilding until finalize()."""
        self.add_documents({chunk_id: text}, rebuild=rebuild)

    def add_documents(self, documents: dict[str, str], *, rebuild: bool = True) -> None:
        """Add sources together; pass ``rebuild=False`` to batch incremental additions."""
        if self.nlp is None:
            self.nlp = load_nlp(self.spacy_model)
        changed = False
        for source_name, text in documents.items():
            cleaned = clean_source_text(text)
            document_id = stable_id("doc", source_name, cleaned)
            if document_id in self._source_documents:
                continue
            self._source_documents[document_id] = (source_name, text)
            changed = True
        if changed:
            self._dirty = True
        if changed and rebuild:
            self._rebuild_corpus()

    def finalize(self) -> None:
        """Build the graph once after one or more deferred source additions."""
        if self._source_documents and self._dirty:
            self._rebuild_corpus()

    def _rebuild_corpus(self) -> None:
        """Resolve aliases corpus-wide before building any document's entity nodes."""
        if not self._source_documents:
            return
        sources = list(self._source_documents.values())
        cleaned_texts = [clean_source_text(text) for _, text in sources]
        corpus_aliases, corpus_alias_displays = discover_corpus_aliases(
            cleaned_texts, self.nlp,
        )

        self.state = None
        self.graph = nx.MultiDiGraph()
        parsed_texts = [mask_heading_text(text) for text in cleaned_texts]
        parsed_documents = self.nlp.pipe(parsed_texts, batch_size=8)
        for (source_name, text), cleaned, parsed_doc in zip(
            sources, cleaned_texts, parsed_documents
        ):
            incoming = build_graph(
                text, source_name, self.spacy_model, self.max_chunk_chars,
                nlp=self.nlp, corpus_aliases=corpus_aliases,
                corpus_alias_displays=corpus_alias_displays,
                parsed_doc=parsed_doc,
            )
            if self.state is None:
                self.state = incoming
                self.graph = incoming["graph"]
                continue

            for node, attributes in incoming["graph"].nodes(data=True):
                if node not in self.graph:
                    self.graph.add_node(node, **attributes)
                    continue
                existing = self.graph.nodes[node]
                for field in ("aliases", "claim_ids", "supporting_chunk_ids", "paragraph_indices"):
                    existing[field].update(attributes[field])
                for field in ("mentions",):
                    existing[field].extend(attributes[field])
                if not existing.get("ent_type") and attributes.get("ent_type"):
                    existing["ent_type"] = attributes["ent_type"]
                    existing["node_type"] = "entity"

            self.graph.add_edges_from(incoming["graph"].edges(keys=True, data=True))
            for field in ("documents", "chunks", "claims", "explicit_aliases"):
                self.state[field].update(incoming[field])
            for field in ("context_bridges", "rejected_triples"):
                self.state[field].extend(incoming[field])

        if len(sources) > 1:
            # Add cross-document navigation bridges and rebuild the backbone once,
            # after all source graphs have been merged.
            self.state["context_bridges"].extend(ensure_connected(
                self.graph, self.state["topic_node"], "corpus", {}
            ))
            self.state["traversal_graph"] = build_traversal_backbone(
                self.graph, self.state["topic_node"], component_count=1
            )
            self.state.pop("source_text", None)
            self.state["source_name"] = "document collection"

        self.state["explicit_aliases"] = corpus_aliases
        component_count = (
            1 if len(sources) > 1 else self.state["diagnostics"]["component_count"]
        )
        self.state["diagnostics"].update(graph_diagnostics(
            self.graph, self.state["traversal_graph"], self.state["claims"],
            self.state["context_bridges"], self.state["rejected_triples"],
            component_count=component_count,
            traversal_component_count=component_count,
        ))
        self._dirty = False

    def save(self, filepath: str = "data/knowledge_graph.pkl") -> None:
        self.finalize()
        if self.state is None:
            raise ValueError("No documents have been indexed")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath: str = "data/knowledge_graph.pkl", spacy_model: str = "en_core_web_sm"):
        builder = cls(spacy_model)
        with Path(filepath).open("rb") as handle:
            builder.state = pickle.load(handle)
        if builder.state.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Legacy graph cache: rebuild it with the Module 2 pipeline")
        builder.graph = builder.state["graph"]
        builder._source_documents = {
            document_id: (document["source_name"], document["text"])
            for document_id, document in builder.state.get("documents", {}).items()
        }
        return builder

    def stats(self) -> dict[str, int]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
        }

    def to_dict(self) -> dict[str, Any]:
        if self.state is None:
            raise ValueError("No documents have been indexed")
        return {**graph_to_dict(self.state), "stats": self.stats()}

def build_graph_from_documents(
    documents: dict[str, str],
    output_path: str | None = "data/knowledge_graph.pkl",
    *,
    spacy_model: str = "en_core_web_sm",
    max_chunk_chars: int = 1200,
) -> KnowledgeGraphBuilder:
    """Index documents with independent source offsets into one connected graph."""
    if not documents:
        raise ValueError("At least one document is required")
    builder = KnowledgeGraphBuilder(spacy_model, max_chunk_chars)
    builder.add_documents(documents)
    if output_path is not None:
        builder.save(output_path)
    return builder
